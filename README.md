# RetailRocket Recommender

Sistema de recomendação Top-K para e-commerce construído a partir do comportamento implícito dos usuários no dataset RetailRocket.

O projeto faz parte do Tech Challenge da Fase 2 da pós-graduação em Machine Learning Engineering e foi desenvolvido com foco em engenharia de software, reprodutibilidade, rastreabilidade de dados, experimentação e comparação entre modelos de recomendação.

## Formulação do problema

Dado o histórico de interações entre usuários e produtos, o sistema deve recomendar os K itens mais relevantes para cada usuário.

O dataset não possui avaliações explícitas, como notas de uma a cinco estrelas. Por isso, o projeto utiliza feedback implícito. Cada evento representa um sinal de interesse com intensidade diferente:

| Evento | Peso |
| --- | ---: |
| `view` | 1.0 |
| `addtocart` | 3.0 |
| `transaction` | 5.0 |

Os pesos são configurados em `params.yaml`, na seção `preprocessing.event_weights`, e não ficam fixados no código de produção.

## Objetivos técnicos

O projeto adota:

- Poetry para gerenciamento de dependências;
- Ruff para lint e formatação;
- Pytest para testes unitários e de integração;
- pre-commit para validações antes dos commits;
- Pydantic Settings para variáveis de ambiente;
- DVC para versionamento de dados, artefatos e pipelines;
- MLflow para tracking de experimentos;
- Scikit-Learn e SciPy para os baselines;
- PyTorch para o modelo neural;
- configuração externa por YAML e `.env`;
- seeds reprodutíveis;
- type hints e docstrings;
- branches separadas por funcionalidade;
- testes para cada nova funcionalidade.

## Estado atual

Os quatro primeiros blocos do projeto estão concluídos:

1. fundação técnica;
2. dataset, DVC inicial, loader e validação;
3. preprocessamento e feature engineering;
4. modelos, treinamento, avaliação, MLflow e integração final com DVC.

O pipeline atual já permite:

- validar o dataset bruto;
- transformar os eventos em feedback implícito;
- gerar interações usuário-item;
- executar split temporal;
- criar encoders;
- gerar negativos para o treino neural;
- treinar um Neural Collaborative Filtering;
- avaliar Popularity, Item-KNN e Neural CF;
- calcular métricas Top-K;
- registrar treinamento e avaliação no MLflow;
- reproduzir todo o fluxo com DVC.

## Estrutura principal

```text
retailrocket-recommender/
├── src/
│   └── retail_recommender/
│       ├── config/
│       │   ├── logging.py
│       │   └── settings.py
│       ├── data/
│       │   ├── loaders/
│       │   ├── preprocessors/
│       │   └── validators/
│       ├── features/
│       │   ├── id_encoder.py
│       │   ├── interaction_builder.py
│       │   ├── negative_sampling.py
│       │   └── temporal_split.py
│       ├── models/
│       │   ├── base.py
│       │   ├── factory.py
│       │   ├── popularity.py
│       │   ├── item_knn.py
│       │   └── neural_cf.py
│       ├── training/
│       │   ├── dataset.py
│       │   ├── early_stopping.py
│       │   ├── seed.py
│       │   └── trainer.py
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── neural_recommender.py
│       │   ├── ranking_metrics.py
│       │   └── reports.py
│       ├── tracking/
│       │   └── mlflow_tracker.py
│       └── pipelines/
│           ├── validate_data.py
│           ├── preprocess.py
│           ├── feature_engineering.py
│           ├── train.py
│           └── evaluate.py
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── training.yaml
│   └── evaluation.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── encoders/
│   ├── models/
│   └── reports/
├── docs/
│   └── architecture.md
├── tests/
│   ├── unit/
│   └── integration/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pyproject.toml
├── poetry.lock
├── .env.example
└── README.md
```

## Requisitos

- Python 3.12;
- Poetry;
- Git;
- DVC.

O projeto foi validado no Windows com Python 3.12.2.

## Instalação

```bash
git clone <URL_DO_REPOSITORIO>
cd retailrocket-recommender
poetry install
```

Crie o arquivo local de ambiente:

```bash
cp .env.example .env
```

No PowerShell:

```powershell
Copy-Item .env.example .env
```

## Variáveis de ambiente

As variáveis operacionais ficam em `.env`.

Exemplo:

```env
APP_NAME=retailrocket-recommender
APP_ENV=local
LOG_LEVEL=INFO
DATA_DIR=data
RAW_DATA_DIR=data/raw
INTERIM_DATA_DIR=data/interim
PROCESSED_DATA_DIR=data/processed
ARTIFACTS_DIR=artifacts
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_EXPERIMENT_NAME=retailrocket-recommender
```

O `.env.example` é versionado. O `.env` local não deve ser versionado.

A random seed não é mais variável de ambiente. Sua fonte única é:

```text
params.yaml → training.random_seed
```

## Configuração centralizada

O projeto separa caminhos operacionais de parâmetros experimentais.

### `params.yaml`

Contém parâmetros que alteram comportamento, treino ou avaliação e que devem ser rastreados pelo DVC.

Principais seções:

```yaml
data:
  minimum_interactions: 10000
  minimum_users: 100
  minimum_items: 100

preprocessing:
  strategy: implicit_feedback
  allowed_event_types:
    - view
    - addtocart
    - transaction
  event_weights:
    view: 1.0
    addtocart: 3.0
    transaction: 5.0

interaction_filtering:
  minimum_user_interactions: 2
  minimum_item_interactions: 2

split:
  strategy: temporal
  train_size: 0.70
  validation_size: 0.15
  test_size: 0.15
  filter_unknown_entities: true

training:
  random_seed: 1729
  negative_samples_per_positive: 4
  batch_size: 1024
  learning_rate: 0.001
  epochs: 30
  patience: 5
  minimum_delta: 0.0
  weight_decay: 0.0
  device: auto

model:
  name: neural_cf
  embedding_dim: 32
  hidden_layers:
    - 64
    - 32
  dropout: 0.2

evaluation:
  k: 10
  candidate_batch_size: 4096
  exclude_seen_items: true
  maximum_users: 50

item_knn:
  n_neighbors: 50
  minimum_similarity: 0.0
```

As antigas chaves `primary_metric` e `metrics` foram removidas porque não eram consumidas pelo pipeline e duplicavam o K nos nomes das métricas.

### `configs/data.yaml`

É a fonte única para caminhos de datasets, encoders e relatórios das etapas de dados.

Principais caminhos:

```text
data/raw/events.csv
data/interim/events_clean.parquet
data/processed/train.parquet
data/processed/train_positive.parquet
data/processed/validation.parquet
data/processed/test.parquet
artifacts/encoders/user_encoder.pkl
artifacts/encoders/item_encoder.pkl
artifacts/reports/data_validation.json
artifacts/reports/feature_engineering_report.json
```

### `configs/model.yaml`

É a fonte única do checkpoint:

```yaml
model:
  checkpoint_path: artifacts/models/best_model.pt
```

### `configs/training.yaml`

É a fonte única para o relatório de treino:

```yaml
training:
  metrics_report_path: artifacts/reports/train_metrics.json
```

### `configs/evaluation.yaml`

É a fonte única para os outputs da avaliação:

```yaml
evaluation:
  output_directory: artifacts/reports/evaluation
```

Essa divisão evita repetir o mesmo caminho em vários YAMLs.

## Dataset

O projeto utiliza o arquivo:

```text
data/raw/events.csv
```

Colunas esperadas:

| Coluna | Descrição |
| --- | --- |
| `timestamp` | instante do evento em Unix timestamp, em milissegundos |
| `visitorid` | identificador do usuário |
| `event` | tipo do evento |
| `itemid` | identificador do produto |
| `transactionid` | identificador da transação, quando houver |

Eventos aceitos:

```text
view
addtocart
transaction
```

O arquivo bruto é versionado pelo DVC, e não diretamente pelo Git.

Para recuperar os dados:

```bash
poetry run dvc pull
```

## Pipeline DVC

O DAG atual contém cinco stages:

```text
validate_data
      ↓
preprocess
      ↓
feature_engineering
      ↓
train
      ↓
evaluate
```

O stage `evaluate` também depende diretamente dos artefatos de feature engineering usados pelos baselines e do checkpoint produzido no treino.

Para visualizar o DAG:

```bash
poetry run dvc dag
```

Para reproduzir toda a cadeia:

```bash
poetry run dvc repro
```

Para verificar o status:

```bash
poetry run dvc status
```

Resultado esperado:

```text
Data and pipelines are up to date.
```

## Stage `validate_data`

Responsabilidades:

- carregar `events.csv`;
- validar colunas obrigatórias;
- validar eventos permitidos;
- verificar mínimos de interações, usuários e itens;
- validar conversão do timestamp;
- produzir relatório estruturado.

Output:

```text
artifacts/reports/data_validation.json
```

## Stage `preprocess`

Responsabilidades:

- padronizar nomes de colunas;
- converter IDs e timestamp;
- remover registros inválidos;
- filtrar eventos suportados;
- atribuir pesos configuráveis;
- gerar coluna temporal;
- ordenar os eventos.

Output:

```text
data/interim/events_clean.parquet
```

Colunas:

```text
user_id
item_id
event_type
event_weight
timestamp
datetime
```

## Stage `feature_engineering`

Responsabilidades:

1. agregar eventos por usuário e item;
2. calcular `interaction_score`;
3. calcular `interaction_count`;
4. obter `last_interaction_at`;
5. filtrar entidades pouco frequentes;
6. executar split temporal;
7. ajustar encoders somente no treino;
8. remover desconhecidos de validação e teste;
9. persistir o treino positivo;
10. gerar negativos para o treino neural;
11. embaralhar o treino;
12. persistir encoders e relatório.

Outputs:

```text
data/processed/train_positive.parquet
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet

artifacts/encoders/user_encoder.pkl
artifacts/encoders/item_encoder.pkl
artifacts/reports/feature_engineering_report.json
```

### `train_positive.parquet`

Contém apenas interações positivas agregadas do período de treino.

É usado por:

- Popularity;
- Item-KNN;
- construção do histórico de itens vistos;
- exclusão de itens já consumidos durante avaliação.

Colunas:

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

O `target` é igual a 1.

### `train.parquet`

Contém o dataset usado pelo Neural CF.

Colunas:

```text
user_idx
item_idx
target
```

Combina:

- positivos observados;
- negativos amostrados.

### `validation.parquet` e `test.parquet`

Mantêm apenas interações positivas reais.

Colunas:

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

## Split temporal

As interações agregadas são ordenadas por `last_interaction_at`.

Divisão atual:

```text
70% treino
15% validação
15% teste
```

A escolha aproxima o cenário de produção, no qual o modelo aprende com o passado e é avaliado no futuro.

Essa abordagem reduz vazamento temporal, mas pode revelar:

- sazonalidade;
- drift;
- mudanças de catálogo;
- mudanças no comportamento dos usuários.

## Cold start

Os encoders são ajustados exclusivamente no treino.

Usuários e itens desconhecidos em validação ou teste recebem temporariamente o índice `-1` e são removidos dos conjuntos finais quando `split.filter_unknown_entities` está habilitado.

A quantidade removida é registrada no relatório de feature engineering.

A avaliação atual mede apenas usuários e itens conhecidos no treino.

## Negative sampling

Uma interação observada é tratada como positiva:

```text
target = 1
```

Como a ausência de interação não é uma rejeição explícita, os negativos são pseudo-negativos gerados entre pares usuário-item não observados:

```text
target = 0
```

Configuração atual:

```text
4 negativos por positivo
```

Fonte única:

```text
params.yaml → training.negative_samples_per_positive
```

A implementação utiliza amostragem por rejeição e nunca converte um par positivo conhecido em negativo.

Os negativos são gerados apenas para o conjunto de treino neural.

## Modelos

### Popularity

Baseline não personalizado.

O modelo ordena itens pela popularidade acumulada nas interações positivas de treino.

Vantagens:

- simples;
- rápido;
- fácil de interpretar;
- importante como referência mínima.

Limitações:

- recomenda praticamente os mesmos itens para todos;
- tende a favorecer itens já muito populares;
- possui baixa cobertura.

### Item-KNN

Baseline baseado em similaridade item-item.

O modelo:

1. constrói uma matriz esparsa usuário-item;
2. calcula vizinhos por similaridade de cosseno;
3. combina o histórico do usuário com os itens similares;
4. filtra vizinhos abaixo de `minimum_similarity`;
5. retorna os itens com maior score.

Parâmetros:

```text
params.yaml → item_knn.n_neighbors
params.yaml → item_knn.minimum_similarity
```

### Neural Collaborative Filtering

Modelo implementado em PyTorch.

Entradas:

```text
user_idx
item_idx
```

Estrutura:

- embedding de usuários;
- embedding de itens;
- concatenação dos embeddings;
- camadas densas;
- dropout;
- saída escalar como logit.

Parâmetros:

```text
embedding_dim
hidden_layers
dropout
```

O treino usa:

- `BCEWithLogitsLoss`;
- Adam;
- early stopping;
- checkpoint da melhor época;
- seed global reprodutível.

## Stage `train`

Entradas:

```text
data/processed/train.parquet
data/processed/validation.parquet
params.yaml
configs/data.yaml
configs/model.yaml
configs/training.yaml
```

Outputs:

```text
artifacts/models/best_model.pt
artifacts/reports/train_metrics.json
```

O relatório registra:

- nome do modelo configurado;
- quantidade de usuários;
- quantidade de itens;
- arquitetura;
- melhor época;
- melhor loss de validação;
- quantidade de épocas concluídas;
- indicação de early stopping;
- caminho do checkpoint.

A primeira execução real apresentou melhor época igual a 1 e aumento posterior da loss de validação, indicando overfitting inicial. Esse comportamento deverá orientar experimentos futuros de regularização e tuning.

## Stage `evaluate`

Modelos avaliados:

```text
popularity
item_knn
neural_cf
```

Entradas principais:

```text
data/processed/train_positive.parquet
data/processed/test.parquet
artifacts/models/best_model.pt
artifacts/reports/train_metrics.json
```

O pipeline:

1. carrega o treino positivo;
2. carrega o teste;
3. carrega metadados do treinamento;
4. reconstrói o Neural CF com dimensões compatíveis com o checkpoint;
5. ajusta Popularity e Item-KNN no treino positivo;
6. cria o histórico de itens vistos;
7. limita usuários quando `maximum_users` está configurado;
8. gera recomendações Top-K;
9. calcula as métricas;
10. salva relatórios individuais e comparação.

Outputs:

```text
artifacts/reports/evaluation/popularity_metrics.json
artifacts/reports/evaluation/item_knn_metrics.json
artifacts/reports/evaluation/neural_cf_metrics.json
artifacts/reports/evaluation/model_comparison.csv
```

## Métricas

### Precision@K

Proporção dos K itens recomendados que são relevantes.

### Recall@K

Proporção dos itens relevantes do usuário recuperados nas recomendações.

### NDCG@K

Mede qualidade do ranking, atribuindo maior peso aos acertos nas primeiras posições.

### MAP@K

Média da precisão acumulada nos pontos em que itens relevantes aparecem.

### Coverage@K

Proporção do catálogo que aparece nas recomendações.

O valor de K é configurado exclusivamente em:

```text
params.yaml → evaluation.k
```

## Limitação de usuários na avaliação

A configuração atual usa:

```text
maximum_users: 50
```

Esse limite reduz custo computacional durante desenvolvimento, especialmente no Neural CF, que pontua candidatos do catálogo para cada usuário.

Por isso, os resultados atuais devem ser interpretados como validação operacional da pipeline, e não como benchmark definitivo.

Comparações entre runs devem usar o mesmo valor de `maximum_users`.

## MLflow

O tracking utiliza:

```text
sqlite:///mlflow.db
```

Experimento padrão:

```text
retailrocket-recommender
```

A configuração fica em `Settings`, alimentada pelas variáveis:

```text
MLFLOW_TRACKING_URI
MLFLOW_EXPERIMENT_NAME
```

O `MLflowTracker` apenas consome essas configurações.

Runs esperadas:

```text
neural_cf_train
popularity_evaluation
item_knn_evaluation
neural_cf_evaluation
model_comparison
```

O treinamento registra:

- hiperparâmetros;
- loss de treino;
- loss de validação;
- histórico por época;
- checkpoint;
- relatório de treinamento.

A avaliação registra:

- K;
- quantidade de usuários avaliados;
- Precision@K;
- Recall@K;
- NDCG@K;
- MAP@K;
- Coverage@K;
- relatórios como artefatos.

Para abrir a interface:

```bash
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Acesse:

```text
http://127.0.0.1:5000
```

Na interface, selecione `Model training`.

## Reprodutibilidade

A seed é configurada somente em:

```text
params.yaml → training.random_seed
```

Ela é utilizada em:

- NumPy;
- negative sampling;
- embaralhamento;
- PyTorch;
- DataLoader;
- avaliação.

O `dvc.lock` registra os parâmetros efetivamente usados em cada execução.

## Testes e qualidade

Validação completa:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run pre-commit run --all-files
poetry check
poetry run dvc repro
poetry run dvc status
```

Testes de pipelines:

```bash
poetry run pytest tests/integration/pipelines -v
```

Testes de modelos:

```bash
poetry run pytest tests/unit/models -v
```

Testes de avaliação:

```bash
poetry run pytest tests/unit/evaluation tests/integration/evaluation -v
```

Testes de tracking:

```bash
poetry run pytest tests/unit/tracking tests/integration/tracking -v
```

## Execuções diretas

Validação:

```bash
poetry run python -m retail_recommender.pipelines.validate_data
```

Preprocessamento:

```bash
poetry run python -m retail_recommender.pipelines.preprocess
```

Feature engineering:

```bash
poetry run python -m retail_recommender.pipelines.feature_engineering
```

Treinamento:

```bash
poetry run python -m retail_recommender.pipelines.train
```

Avaliação:

```bash
poetry run python -m retail_recommender.pipelines.evaluate
```

Para reprodutibilidade, prefira `poetry run dvc repro`.

## Blocos concluídos

### Bloco 1 — Fundação técnica

Entregas principais:

- estrutura inicial;
- Poetry;
- Ruff;
- Pytest;
- pre-commit;
- Pydantic Settings;
- logging;
- script de validação;
- Makefile;
- testes iniciais.

### Bloco 2 — Dataset e DVC inicial

Entregas principais:

- RetailRocket;
- DVC;
- remote local;
- loader;
- validador;
- pipeline `validate_data`;
- relatório de validação;
- testes.

### Bloco 3 — Preparação dos dados

Entregas principais:

- Strategy para preprocessamento;
- feedback implícito;
- pesos de eventos;
- agregação usuário-item;
- split temporal;
- encoders;
- cold start;
- negative sampling;
- `train_positive.parquet`;
- `train.parquet`;
- validação e teste;
- stages `preprocess` e `feature_engineering`.

### Bloco 4 — Modelagem, tracking e avaliação

Entregas principais:

- interface base dos recomendadores;
- factory de modelos;
- Popularity;
- Item-KNN;
- Neural Collaborative Filtering;
- dataset e DataLoader;
- early stopping;
- checkpoint;
- treinamento reprodutível;
- MLflow;
- métricas Top-K;
- avaliação dos três modelos;
- comparação de resultados;
- stages DVC `train` e `evaluate`;
- centralização das configurações;
- testes unitários e de integração.

## Limitações atuais

- cold start não resolvido;
- dados de conteúdo dos produtos ainda não utilizados;
- avaliação limitada por `maximum_users`;
- Neural CF apresentou overfitting inicial;
- sem tuning sistemático;
- sem hard negative sampling;
- sem avaliação walk-forward;
- sem retreinamento por janela móvel;
- sem Model Registry efetivamente configurado;
- sem Docker e CI/CD;
- possível presença de falsos negativos entre itens não observados.

## Próximos passos

As próximas etapas previstas incluem:

1. tuning e regularização do Neural CF;
2. avaliação em amostra maior ou conjunto completo;
3. análise comparativa das runs;
4. seleção do melhor modelo;
5. Model Registry;
6. empacotamento com Docker;
7. automação de qualidade e CI/CD;
8. documentação final e roteiro do vídeo.

## Conclusão

O projeto possui agora uma cadeia reprodutível de ponta a ponta, desde o dado bruto até a comparação de modelos.

Os artefatos de dados são versionados pelo DVC, os parâmetros experimentais possuem fonte única em `params.yaml`, os caminhos são separados por responsabilidade nos arquivos de `configs/`, o treinamento e a avaliação são rastreados pelo MLflow, e os modelos são comparados por métricas de ranking Top-K.

O Bloco 4 encerra a primeira implementação completa do sistema de recomendação e deixa uma base consistente para tuning, governança de modelos, containerização e automação.
