# Arquitetura do RetailRocket Recommender

## 1. Visão geral

O sistema implementa uma arquitetura modular para recomendação Top-K com feedback implícito.

O fluxo completo é:

```text
events.csv
    ↓
validate_data
    ↓
events_clean.parquet
    ↓
feature_engineering
    ├── train_positive.parquet
    ├── train.parquet
    ├── validation.parquet
    ├── test.parquet
    ├── user_encoder.pkl
    └── item_encoder.pkl
          ↓
        train
          ├── best_model.pt
          └── train_metrics.json
                ↓
              evaluate
                ├── popularity_metrics.json
                ├── item_knn_metrics.json
                ├── neural_cf_metrics.json
                ├── model_comparison.csv
                └── selected_model.json
                          ↓
                    register_model
                          ├── MLflow run
                          ├── Registered Model
                          ├── alias staging
                          └── model_registration.json
```

A arquitetura separa:

- ingestão e validação;
- preprocessamento;
- feature engineering;
- modelos;
- treinamento;
- avaliação;
- tracking;
- configuração;
- versionamento de artefatos.

## 2. Princípios arquiteturais

O projeto segue os seguintes princípios:

- responsabilidade única por módulo;
- dependências explícitas;
- configuração externa;
- fonte única para cada parâmetro ou caminho;
- testes unitários e de integração;
- artefatos reproduzíveis;
- separação entre dados positivos e treino neural;
- abstrações comuns para recomendadores;
- rastreabilidade por DVC e MLflow.

## 3. Camada de configuração

### 3.1 Variáveis de ambiente

`src/retail_recommender/config/settings.py` centraliza configurações operacionais.

Exemplos:

```text
APP_NAME
APP_ENV
LOG_LEVEL
DATA_DIR
RAW_DATA_DIR
INTERIM_DATA_DIR
PROCESSED_DATA_DIR
ARTIFACTS_DIR
MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME
MLFLOW_REGISTERED_MODEL_NAME
MLFLOW_STAGING_ALIAS
MLFLOW_PRODUCTION_ALIAS
```

O backend padrão do MLflow é:

```text
sqlite:///mlflow.db
```

O tracker não lê diretamente `os.getenv`. Ele consome `Settings`.

### 3.2 Parâmetros experimentais

`params.yaml` é a fonte única para:

- pesos dos eventos;
- filtros mínimos;
- split temporal;
- random seed;
- negative sampling;
- hiperparâmetros do treino;
- arquitetura do Neural CF;
- parâmetros do Item-KNN;
- parâmetros da avaliação.

A seed existe somente em:

```text
training.random_seed
```

O K existe somente em:

```text
evaluation.k
```

### 3.3 Caminhos

Responsabilidades:

```text
configs/data.yaml
→ datasets, encoders e relatórios de dados

configs/model.yaml
→ checkpoint

configs/training.yaml
→ relatório de treino

configs/evaluation.yaml
→ diretório de outputs da avaliação

configs/registry.yaml
→ relatório de registro do modelo
```

Essa separação elimina caminhos repetidos em múltiplos YAMLs.

## 4. Camada de dados

### 4.1 Loader

O loader:

- verifica existência;
- carrega o CSV;
- rejeita arquivo vazio;
- não executa transformação de negócio.

### 4.2 Validador

O validador verifica:

- colunas;
- dados ausentes;
- eventos;
- volume mínimo;
- usuários;
- itens;
- timestamp.

Output:

```text
artifacts/reports/data_validation.json
```

### 4.3 Preprocessador

`ImplicitFeedbackPreprocessor` implementa uma estratégia de preprocessamento.

Entradas:

```text
event_weights
allowed_event_types
```

Ele não conhece valores fixos de peso. Os valores vêm de `params.yaml`.

Output:

```text
data/interim/events_clean.parquet
```

## 5. Construção das interações

Eventos são agregados por:

```text
user_id
item_id
```

Campos produzidos:

```text
interaction_score
interaction_count
last_interaction_at
target
```

A pontuação é a soma dos pesos dos eventos.

O `target` inicial é 1 porque cada linha agregada representa interação observada.

## 6. Split temporal

As interações são ordenadas por `last_interaction_at`.

Divisão:

```text
70% treino
15% validação
15% teste
```

A divisão é global no tempo.

Benefício:

- reduz vazamento temporal.

Riscos:

- sazonalidade;
- drift;
- mudança de catálogo;
- mudança de perfil de consumo.

## 7. Encoders e cold start

Os encoders transformam IDs esparsos em índices contínuos:

```text
user_id → user_idx
item_id → item_idx
```

Eles são ajustados apenas no treino.

Entidades desconhecidas recebem temporariamente `-1`.

Quando `filter_unknown_entities` está habilitado, linhas desconhecidas são removidas de validação e teste.

Essa política garante compatibilidade com embeddings, mas limita a avaliação a entidades conhecidas.

## 8. Positive e negative sampling

### 8.1 Positivos

Interações observadas:

```text
target = 1
```

O arquivo:

```text
data/processed/train_positive.parquet
```

preserva as interações positivas agregadas e suas colunas completas.

Ele é usado pelos baselines e pelo histórico de itens vistos.

### 8.2 Negativos

Pares não observados são amostrados como pseudo-negativos:

```text
target = 0
```

Configuração:

```text
training.negative_samples_per_positive
```

Valor atual:

```text
4
```

A amostragem é por rejeição:

1. sorteia item conhecido;
2. rejeita se já for positivo do usuário;
3. aceita caso contrário;
4. repete até atingir a quantidade necessária.

O pipeline valida que negativos não se sobrepõem a positivos.

### 8.3 Treino neural

`train.parquet` contém somente:

```text
user_idx
item_idx
target
```

Ele combina positivos e negativos e é usado pelo Neural CF.

## 9. Camada de modelos

### 9.1 Interface base

Os recomendadores compartilham um contrato comum para:

- `fit`;
- `recommend`;
- persistência quando aplicável;
- validação de entrada.

A factory normaliza nomes e cria implementações suportadas.

### 9.2 Popularity

Ordena itens por score agregado de popularidade.

Características:

- não personalizado;
- baixo custo;
- alta interpretabilidade;
- referência mínima.

### 9.3 Item-KNN

Constrói matriz usuário-item esparsa e vizinhança item-item por cosseno.

Fluxo:

1. matriz esparsa;
2. busca de vizinhos;
3. remoção do próprio item;
4. aplicação de `minimum_similarity`;
5. score ponderado pelo histórico do usuário;
6. ordenação dos candidatos.

Parâmetros:

```text
item_knn.n_neighbors
item_knn.minimum_similarity
```

### 9.4 Neural CF

Arquitetura:

```text
user_idx → user embedding
item_idx → item embedding
embeddings concatenados
        ↓
MLP
        ↓
logit
```

Parâmetros:

```text
model.embedding_dim
model.hidden_layers
model.dropout
```

A dimensão dos embeddings depende do número de usuários e itens conhecidos.

## 10. Camada de treinamento

Componentes:

- dataset PyTorch;
- DataLoader;
- seed global;
- trainer;
- early stopping;
- checkpoint;
- relatório.

Loss:

```text
BCEWithLogitsLoss
```

Otimizador:

```text
Adam
```

O checkpoint corresponde à melhor loss de validação, não necessariamente à última época.

O relatório guarda as dimensões das entidades para reconstrução posterior do modelo.

## 11. Camada de avaliação

### 11.1 Entradas

```text
train_positive.parquet
test.parquet
best_model.pt
train_metrics.json
```

### 11.2 Histórico de itens vistos

O histórico é construído a partir de `train_positive.parquet`.

Quando `exclude_seen_items` é verdadeiro, itens conhecidos do treino são removidos das recomendações.

### 11.3 Candidatos

O Neural CF pontua itens em lotes de tamanho:

```text
evaluation.candidate_batch_size
```

Isso reduz uso de memória.

### 11.4 Usuários avaliados

A quantidade pode ser limitada por:

```text
evaluation.maximum_users
```

O valor atual de 50 é adequado para smoke tests e desenvolvimento, mas não representa avaliação definitiva.

### 11.5 Métricas

O evaluator calcula:

```text
Precision@K
Recall@K
NDCG@K
MAP@K
Coverage@K
```

O cutoff é definido somente em `evaluation.k`.

## 12. MLflow

### 12.1 Backend

```text
sqlite:///mlflow.db
```

### 12.2 Experimento

```text
retailrocket-recommender
```

### 12.3 Runs

```text
neural_cf_train
popularity_evaluation
item_knn_evaluation
neural_cf_evaluation
model_comparison
```

### 12.4 Tracking do treino

Registra:

- seed;
- batch size;
- learning rate;
- arquitetura;
- dropout;
- épocas;
- early stopping;
- losses;
- checkpoint;
- relatório.

### 12.5 Tracking da avaliação

Registra:

- modelo;
- K;
- usuários;
- exclusão de vistos;
- parâmetros do modelo;
- métricas;
- relatórios.


## 13. Seleção de modelos

`evaluation/model_selector.py` recebe o relatório consolidado e seleciona o melhor modelo pela métrica primária.

A configuração atual utiliza NDCG@10.

Critérios de desempate:

```text
Recall@10
MAP@10
Coverage@10
nome do modelo
```

O último critério garante comportamento determinístico mesmo quando todas as métricas anteriores são iguais.

Output:

```text
artifacts/reports/evaluation/selected_model.json
```

O componente não conhece detalhes de MLflow. Sua responsabilidade termina na decisão baseada em métricas e na persistência da seleção.

## 14. Model Registry

`pipelines/register_model.py` orquestra o registro.

`tracking/registry.py` encapsula:

- criação do modelo registrado;
- criação de versões;
- consulta por alias;
- atribuição de aliases;
- tradução de erros do MLflow.

`tracking/item_knn_pyfunc.py` adapta o Item-KNN ao contrato `mlflow.pyfunc.PythonModel`.

Fluxo:

```text
selected_model.json
        ↓
register_model.py
        ↓
ItemKNNPyFunc
        ↓
MLflow run
        ↓
Registered Model
        ↓
alias staging
```

O nome padrão do modelo registrado é:

```text
RetailRocketRecommender
```

O relatório final é salvo em:

```text
artifacts/reports/registry/model_registration.json
```

O alias `production` existe na configuração, mas não é atribuído automaticamente.

## 15. Arquitetura Docker

O Dockerfile possui dois ambientes finais:

```text
runtime
→ aplicação, dependências principais e MLflow

pipeline
→ aplicação, DVC, testes e ferramentas de qualidade
```

O Poetry fica em:

```text
/opt/poetry
```

As dependências do projeto ficam em:

```text
/opt/venv
```

Essa separação impede que operações de instalação das dependências do projeto removam o próprio Poetry.

A ordem do `PATH` prioriza `/opt/venv/bin`, garantindo que Python, pip, DVC, Pytest e Ruff usem o ambiente da aplicação.

A imagem final não inclui:

- `.git`;
- virtualenv local;
- dados brutos;
- cache do DVC;
- artefatos gerados;
- caches de ferramentas.

## 16. Arquitetura do Docker Compose

```text
host
│
├── navegador
│      ↓ localhost:5000
│
└── Docker Compose network
       ├── mlflow
       │     ├── SQLite
       │     └── artifact store
       │
       └── trainer
             ├── /app montado do host
             ├── DVC
             ├── pipeline
             └── MLFLOW_TRACKING_URI=http://mlflow:5000
```

O serviço `mlflow` fornece tracking, interface web, backend SQLite, artifact store e Model Registry.

O serviço `trainer` usa a imagem `pipeline`, monta o repositório em `/app` e acessa o MLflow pelo hostname interno `mlflow`.

O volume nomeado persiste:

```text
/mlflow/mlflow.db
/mlflow/artifacts
```

Esses dados permanecem após `docker compose down`.

## 17. Makefile como interface operacional

O Makefile centraliza:

```text
instalação
qualidade
testes
DVC
MLflow local
Docker
Docker Compose
Model Registry
validação final
```

Exemplos:

```bash
make quality
make pipeline
make compose-up
make compose-pipeline
make compose-register-model
make validate-local
```

A camada operacional não contém regras de negócio. Ela apenas padroniza comandos já existentes.

## 18. Fronteiras arquiteturais

```text
evaluation/model_selector.py
→ decisão baseada em métricas

tracking/registry.py
→ integração com o Model Registry

tracking/item_knn_pyfunc.py
→ adaptação para MLflow PyFunc

pipelines/register_model.py
→ orquestração do caso de uso

Dockerfile
→ construção dos ambientes

docker-compose.yml
→ composição, rede e persistência

Makefile
→ interface operacional
```

## 19. Fluxo final de entrega

```text
código + parâmetros + dados versionados
                ↓
            dvc repro
                ↓
       comparação de modelos
                ↓
        seleção automática
                ↓
        registro no MLflow
                ↓
          alias staging
                ↓
     validações local e Docker
```

## 20. DVC

Stages:

```text
validate_data
preprocess
feature_engineering
train
evaluate
```

O DVC rastreia:

- código;
- configurações;
- parâmetros;
- dados;
- relatórios;
- checkpoint.

O `dvc.lock` registra o estado efetivamente executado.

A reprodução completa é:

```bash
poetry run dvc repro
```

## 21. Contratos dos principais artefatos

### `events_clean.parquet`

```text
user_id
item_id
event_type
event_weight
timestamp
datetime
```

### `train_positive.parquet`

```text
user_id
item_id
user_idx
item_idx
interaction_score
interaction_count
last_interaction_at
target
```

### `train.parquet`

```text
user_idx
item_idx
target
```

### `validation.parquet` e `test.parquet`

```text
user_id
item_id
user_idx
item_idx
interaction_score
interaction_count
last_interaction_at
target
```

### `train_metrics.json`

Inclui:

- resultado do treino;
- melhor época;
- melhor loss;
- dimensões de usuários e itens;
- arquitetura;
- checkpoint.

### Relatórios de avaliação

Cada JSON contém:

- nome do modelo;
- parâmetros;
- métricas;
- metadados da execução.

O CSV consolida os modelos para comparação.

## 22. Decisões arquiteturais

Decisões atuais:

- feedback implícito;
- pesos por evento;
- split temporal global;
- encoders treinados somente no treino;
- remoção de desconhecidos;
- quatro negativos por positivo;
- baselines Popularity e Item-KNN;
- Neural CF com embeddings e MLP;
- BCEWithLogitsLoss;
- early stopping;
- checkpoint da melhor época;
- avaliação Top-K;
- exclusão opcional de itens vistos;
- MLflow local ou em Docker Compose;
- seleção automática pelo NDCG@10;
- Model Registry com alias `staging`;
- empacotamento Item-KNN como MLflow PyFunc;
- DVC para o pipeline completo;
- Docker multi-stage;
- persistência do MLflow em volume nomeado;
- parâmetros centralizados;
- caminhos centralizados por responsabilidade;
- comandos operacionais centralizados no Makefile.

## 23. Limitações

- cold start não resolvido;
- sem features de conteúdo;
- sem contexto temporal explícito no modelo;
- sem hard negative sampling;
- falsos negativos possíveis;
- avaliação limitada a parte dos usuários;
- custo de ranking sobre catálogo grande;
- overfitting inicial do Neural CF;
- sem tuning sistemático;
- sem walk-forward;
- sem janela móvel;
- cobertura baixa do Item-KNN;
- sem serving de produção;
- sem promoção automática para `production`;
- sem monitoramento de drift;
- sem métricas online.

## 24. Evoluções previstas

- tuning de hiperparâmetros;
- regularização;
- avaliação sobre amostra maior;
- ranking mais eficiente;
- amostragem de candidatos;
- hard negatives;
- features de item;
- features temporais;
- estratégia híbrida para cold start;
- critérios formais para promoção;
- serving;
- monitoramento;
- métricas de diversidade e novidade;
- CI/CD.
