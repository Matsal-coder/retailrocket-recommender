# RetailRocket Recommender

Sistema de recomendação de produtos para e-commerce baseado em comportamento de navegação dos usuários.

Este projeto faz parte do Tech Challenge da Fase 2 da pós-graduação em Machine Learning Engineering e utiliza o dataset RetailRocket, principalmente o arquivo `events.csv`, com eventos de interação em e-commerce.

## Formulação do problema

Dado o histórico de interações de usuários com produtos em um e-commerce, construir um modelo capaz de recomendar os Top-K produtos mais relevantes para cada usuário.

A abordagem adotada será de recomendação com feedback implícito, em que diferentes tipos de eventos recebem pesos distintos conforme a força do sinal de interesse do usuário.

Pesos iniciais sugeridos:

| Evento | Peso |
| --- | ---: |
| `view` | 1 |
| `addtocart` | 3 |
| `transaction` | 5 |

## Objetivos técnicos do projeto

O projeto será construído com foco em boas práticas de engenharia de software, MLOps e reprodutibilidade.

Entre os principais requisitos técnicos estão:

- estrutura de projeto com Clean Code;
- gerenciamento de dependências com Poetry;
- dependências de produção e desenvolvimento separadas;
- lock file versionado;
- lint e formatação com Ruff;
- testes automatizados com Pytest;
- hooks de qualidade com pre-commit;
- configuração externa via `.env`;
- logging padronizado;
- versionamento de dados e pipelines com DVC;
- tracking de experimentos e Model Registry com MLflow;
- modelos de recomendação com Scikit-Learn e PyTorch;
- containerização com Docker.

## Estrutura atual do projeto

```text
retailrocket-recommender/
├── src/
│   └── retail_recommender/
│       ├── config/
│       │   ├── settings.py
│       │   └── logging.py
│       ├── data/
│       │   ├── loaders/
│       │   │   └── retailrocket_loader.py
│       │   ├── validators/
│       │   │   └── events_validator.py
│       │   └── preprocessors/
│       ├── features/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── tracking/
│       └── pipelines/
│           └── validate_data.py
├── configs/
│   └── data.yaml
├── data/
│   ├── raw/
│   │   └── events.csv.dvc
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── models/
│   ├── encoders/
│   └── reports/
│       └── data_validation.json
├── docs/
├── tests/
│   ├── unit/
│   │   └── data/
│   │       ├── loaders/
│   │       └── validators/
│   └── integration/
│       └── pipelines/
├── scripts/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pyproject.toml
├── poetry.lock
├── .env.example
├── .gitignore
├── .dockerignore
├── .dvcignore
├── .pre-commit-config.yaml
├── README.md
└── Makefile
```

## Principais diretórios

### `src/retail_recommender/`

Contém o código-fonte principal da aplicação.

### `configs/`

Contém arquivos de configuração em YAML.

Atualmente, o arquivo principal é:

```text
configs/data.yaml
```

Ele define caminhos e parâmetros básicos para validação do dataset.

### `data/`

Contém os dados locais do projeto.

A pasta é dividida em:

- `data/raw/`: dados brutos;
- `data/interim/`: dados intermediários;
- `data/processed/`: dados prontos para modelagem.

Os dados reais não devem ser versionados diretamente no Git.

O arquivo bruto principal esperado é:

```text
data/raw/events.csv
```

Esse arquivo é versionado pelo DVC por meio de:

```text
data/raw/events.csv.dvc
```

### `artifacts/`

Contém artefatos gerados pelo projeto, como relatórios, modelos e encoders.

Após a validação dos dados, o relatório é salvo em:

```text
artifacts/reports/data_validation.json
```

### `tests/`

Contém testes unitários e de integração.

Atualmente existem testes para:

- configurações;
- logging;
- validação de ambiente;
- loader do RetailRocket;
- validador estrutural do `events.csv`;
- pipeline de validação dos dados.

### `scripts/`

Contém scripts auxiliares de desenvolvimento e operação.

## Requisitos

- Python >= 3.11 e < 3.13
- Poetry
- Git
- DVC

## Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd retailrocket-recommender
```

Instale as dependências:

```bash
poetry install
```

Crie o arquivo local de variáveis de ambiente:

```bash
cp .env.example .env
```

No Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

## Configuração via ambiente

As configurações principais são lidas a partir de variáveis de ambiente.

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
RANDOM_SEED=317
```

O arquivo `.env.example` deve ser versionado.

O arquivo `.env` local não deve ser versionado.

## Dataset

O dataset escolhido é o RetailRocket.

Neste projeto, o principal arquivo utilizado é:

```text
events.csv
```

Ele contém eventos de interação entre usuários e produtos, como:

- `view`;
- `addtocart`;
- `transaction`.

O arquivo bruto deve estar localizado em:

```text
data/raw/events.csv
```

## Contrato esperado do `events.csv`

O arquivo `events.csv` deve conter, no mínimo, as seguintes colunas:

| Coluna | Descrição |
| --- | --- |
| `timestamp` | momento do evento em timestamp Unix em milissegundos |
| `visitorid` | identificador do usuário/visitante |
| `event` | tipo de evento realizado |
| `itemid` | identificador do item/produto |
| `transactionid` | identificador da transação, quando houver |

Eventos esperados:

```text
view
addtocart
transaction
```

## Versionamento de dados com DVC

Os dados brutos não são versionados diretamente pelo Git.

O arquivo:

```text
data/raw/events.csv
```

é rastreado pelo DVC por meio do arquivo:

```text
data/raw/events.csv.dvc
```

Após clonar o repositório e instalar as dependências, recupere os dados com:

```bash
poetry run dvc pull
```

Para verificar o status dos dados e pipelines:

```bash
poetry run dvc status
```

Resultado esperado quando tudo está sincronizado:

```text
Data and pipelines are up to date.
```

## Configuração dos dados

As configurações principais da validação estão em:

```text
configs/data.yaml
```

Exemplo:

```yaml
raw_events_path: data/raw/events.csv
validation_report_path: artifacts/reports/data_validation.json

required_columns:
  - timestamp
  - visitorid
  - event
  - itemid
  - transactionid

allowed_events:
  - view
  - addtocart
  - transaction

minimum_interactions: 10000
minimum_users: 100
minimum_items: 100
```

Os parâmetros também são registrados em:

```text
params.yaml
```

Esses parâmetros são usados pelo DVC para rastrear mudanças relevantes no pipeline.

## Loader do RetailRocket

O loader do dataset está implementado em:

```text
src/retail_recommender/data/loaders/retailrocket_loader.py
```

Responsabilidades do loader:

- receber o caminho do arquivo `events.csv`;
- verificar se o arquivo existe;
- carregar o CSV como `pandas.DataFrame`;
- falhar com erro claro caso o arquivo esteja ausente ou vazio.

O loader não realiza preprocessamento, agregação, split temporal ou transformação de feedback implícito.

## Validador do dataset

O validador estrutural está implementado em:

```text
src/retail_recommender/data/validators/events_validator.py
```

A validação verifica:

- existência das colunas obrigatórias;
- se `timestamp`, `visitorid`, `event` e `itemid` não estão totalmente vazios;
- se os eventos pertencem ao conjunto esperado;
- se há pelo menos 10.000 interações no dataset real;
- se é possível converter `timestamp` para datetime;
- se há usuários e itens suficientes para um problema de recomendação;
- se o resultado da validação pode ser convertido para dicionário e salvo em JSON.

O validador retorna um objeto estruturado com:

- `is_valid`;
- `errors`;
- `warnings`;
- `summary`.

## Pipeline de dados e feature engineering

O projeto utiliza DVC para reproduzir o processamento dos dados desde o dataset bruto até os conjuntos preparados para treinamento e avaliação.

O pipeline atual contém três stages principais:

validate_data
      ↓
preprocess
      ↓
feature_engineering

### validate_data

Valida o arquivo bruto events.csv, verificando:

- existência das colunas obrigatórias;
- tipos e valores válidos;
- eventos suportados;
- quantidade mínima de interações;
- quantidade mínima de usuários;
- quantidade mínima de itens.

O stage gera:

artifacts/reports/data_validation.json

### preprocess

Converte os eventos brutos em um formato padronizado para feedback implícito.

As principais transformações são:

- remoção de registros inválidos;
- conversão do timestamp;
- criação da coluna temporal;
- normalização dos nomes das colunas;
- filtragem dos eventos suportados;
- aplicação de pesos por tipo de evento.

Pesos utilizados:

- view: 1.0
- addtocart: 3.0
- transaction: 5.0

O stage gera:

data/interim/events_clean.parquet

### feature_engineering

Constrói os dados necessários para treinamento e avaliação do sistema de recomendação.

O stage executa:

1. agregação das interações por usuário e item;
2. filtragem de usuários e itens pouco frequentes;
3. split temporal em treino, validação e teste;
4. ajuste dos encoders somente no treino;
5. transformação dos IDs originais em índices contínuos;
6. geração de exemplos negativos somente para treino;
7. persistência dos conjuntos processados;
8. persistência dos encoders;
9. geração de relatório de feature engineering.

## Artefatos gerados

Após executar o pipeline completo, os principais outputs são:

data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet

artifacts/encoders/user_encoder.pkl
artifacts/encoders/item_encoder.pkl

artifacts/reports/data_validation.json
artifacts/reports/feature_engineering_report.json

### Conjunto de treino

O arquivo train.parquet contém:

user_idx
item_idx
target

O target assume:

- 1 para interações positivas observadas;
- 0 para pares negativos amostrados.

### Conjuntos de validação e teste

Os arquivos de validação e teste mantêm as colunas necessárias para avaliação posterior:

user_id
item_id
user_idx
item_idx
interaction_score
interaction_count
last_interaction_at
target

Nesta etapa, validação e teste contêm apenas interações positivas reais.

## Split temporal

As interações são ordenadas pela coluna last_interaction_at e divididas em:

- 70% para treino;
- 15% para validação;
- 15% para teste.

O split temporal foi escolhido para reproduzir o cenário de produção, no qual o modelo aprende com o passado e é avaliado em interações posteriores.

Essa abordagem reduz o risco de vazamento temporal, mas pode expor mudanças legítimas de comportamento, sazonalidade e drift.

Por exemplo, produtos sazonais podem ter padrões diferentes entre os períodos de treino e teste. Por isso, o relatório de feature engineering registra os intervalos temporais de cada conjunto.

### Cold start

Os encoders são ajustados exclusivamente com o conjunto de treino.

Usuários e itens que aparecem somente em validação ou teste não possuem índices conhecidos pelo modelo. Nesta versão, essas interações são removidas dos conjuntos de avaliação.

O relatório registra:

validation_removed_unknowns
test_removed_unknowns

Essa política significa que a avaliação inicial cobre apenas usuários e itens conhecidos no treino.

## Encoders de usuários e itens

Os identificadores originais do RetailRocket podem ser esparsos e não contínuos.

Por isso, o pipeline cria índices inteiros contínuos:

user_id → user_idx
item_id → item_idx

Exemplo:

user_id 105 → user_idx 0
user_id 900 → user_idx 1

Os índices começam em zero e são compatíveis com camadas de embedding do PyTorch.

Os encoders são ajustados apenas no treino e persistidos em:

artifacts/encoders/user_encoder.pkl
artifacts/encoders/item_encoder.pkl

Valores desconhecidos recebem temporariamente o índice -1, mas esses valores não são persistidos nos conjuntos finais.

## Negative sampling

O RetailRocket contém feedback implícito.

Uma interação observada indica algum nível de interesse do usuário, mas a ausência de interação não representa rejeição explícita.

Os pares observados são tratados como positivos:

target = 1

Para que o modelo aprenda a distinguir itens relevantes de itens sem evidência de interesse, o pipeline gera pares negativos:

target = 0

Um par negativo é formado por:

- um usuário conhecido;
- um item conhecido;
- ausência de interação positiva entre eles no conjunto de treino.

O pipeline nunca transforma um par positivo conhecido em negativo.

A configuração inicial utiliza quatro negativos para cada positivo.

Essa proporção é um hiperparâmetro e pode ser alterada em params.yaml.

A razão 4:1 foi escolhida como baseline por oferecer mais contraste ao modelo sem aumentar excessivamente o tamanho do conjunto de treino.

Os negativos são gerados somente para treino. Validação e teste mantêm as interações positivas reais para futura avaliação Top-K.

### Desempenho do negative sampling

A implementação utiliza amostragem por rejeição.

Em vez de construir, para cada usuário, uma lista contendo todos os itens disponíveis do catálogo, o algoritmo:

1. sorteia índices diretamente no intervalo de itens conhecidos;
2. rejeita os itens que já são positivos para o usuário;
3. continua até atingir a quantidade configurada de negativos.

Essa abordagem evita percorrer todo o catálogo para cada usuário e reduz significativamente o custo do processamento em datasets grandes.

## Principais parâmetros

Os principais parâmetros do pipeline estão em params.yaml.

Exemplo:

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
  negative_samples_per_positive: 4
  random_seed: <seed configurada no projeto>

A seed é mantida fixa para garantir reprodutibilidade da amostragem negativa e do embaralhamento dos dados de treino.

## Executando o pipeline

Para reproduzir todos os stages:

poetry run dvc repro

Para executar somente o feature engineering:

poetry run dvc repro feature_engineering

Para visualizar o DAG:

poetry run dvc dag

Para verificar se os dados e artefatos estão atualizados:

poetry run dvc status

Para executar o pipeline diretamente, sem o DVC:

poetry run python -m retail_recommender.pipelines.feature_engineering

A execução direta é útil para desenvolvimento e diagnóstico. Para reprodutibilidade, prefira dvc repro.

## Validação do projeto

Antes de realizar commits ou merges, execute:

poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run pre-commit run --all-files
poetry run dvc repro
poetry run dvc status

Essas verificações garantem:

- testes automatizados;
- padronização do código;
- validações do pre-commit;
- reprodutibilidade dos dados;
- atualização do dvc.lock.

## Testes

Para rodar todos os testes:

```bash
poetry run pytest
```

Ou:

```bash
make test
```

Para rodar testes com cobertura:

```bash
poetry run pytest --cov=retail_recommender --cov-report=term-missing
```

Ou:

```bash
make test-cov
```

Para rodar apenas os testes do loader:

```bash
poetry run pytest tests/unit/data/loaders
```

Para rodar apenas os testes do validador:

```bash
poetry run pytest tests/unit/data/validators
```

Para rodar apenas os testes de integração dos pipelines:

```bash
poetry run pytest tests/integration/pipelines
```

## Lint e formatação

Para verificar problemas de lint:

```bash
poetry run ruff check .
```

Ou:

```bash
make lint
```

Para formatar o código:

```bash
poetry run ruff format .
poetry run ruff check . --fix
```

Ou:

```bash
make format
```

## Pre-commit

Instale os hooks:

```bash
poetry run pre-commit install
```

Rode todos os hooks manualmente:

```bash
poetry run pre-commit run --all-files
```

Ou:

```bash
make pre-commit
```

## Validação do ambiente

Rode:

```bash
poetry run python scripts/validate_env.py
```

Ou, se tiver `make` disponível:

```bash
make validate
```

Resultado esperado:

```text
Environment validation finished successfully
```

## Comandos principais de validação do projeto

Antes de abrir Pull Request ou fazer merge, rode:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry run pre-commit run --all-files
poetry run dvc repro
poetry run dvc status
```

Resultado esperado:

- testes passando;
- Ruff sem erros;
- formatação validada;
- hooks de pre-commit passando;
- pipeline DVC completo reproduzível;
- dados e artefatos atualizados.

## Estratégia atual de preparação dos dados

A formulação utiliza recomendação com feedback implícito.

Os eventos recebem os seguintes pesos:

```text
view = 1
addtocart = 3
transaction = 5
```

Esses pesos são efetivamente aplicados no stage `preprocess` e podem ser ajustados em experimentos futuros.

Depois do preprocessamento, os eventos são agregados por par usuário-item. O resultado contém:

```text
interaction_score
interaction_count
last_interaction_at
target
```

O pipeline aplica split temporal global em:

```text
70% treino
15% validação
15% teste
```

Os encoders são ajustados somente no treino. Usuários e itens desconhecidos são removidos de validação e teste e contabilizados no relatório de feature engineering.

O conjunto de treino combina:

```text
interações positivas observadas
+
quatro negativos amostrados por positivo
```

Os negativos são gerados somente entre pares usuário-item sem interação positiva conhecida no treino.

## Reprodutibilidade

O projeto usa seeds fixas para reduzir variação entre execuções.

A seed configurada é:

```text
RANDOM_SEED=317
```

Ela é utilizada nas etapas que exigem aleatoriedade, incluindo:

- NumPy;
- negative sampling;
- embaralhamento do conjunto de treino;
- futuros experimentos com Scikit-Learn;
- futuros modelos em PyTorch.

A reprodução dos dados e artefatos é controlada pelo DVC por meio dos stages:

```text
validate_data
      ↓
preprocess
      ↓
feature_engineering
```

## Status atual

Bloco atual concluído:

```text
Bloco 3 — Preprocessamento, feature engineering e preparação dos dados
```

Status:

```text
Concluído
```

O projeto já possui uma fundação técnica reproduzível e os conjuntos necessários para iniciar a modelagem.

## O que foi concluído até aqui

### Bloco 1 — Estrutura inicial, Clean Code e ambiente base

Objetivo do bloco:

Criar a fundação técnica do projeto, padronizar o ambiente de desenvolvimento e estabelecer as primeiras garantias de qualidade.

Concluído:

- criação da estrutura inicial de pastas;
- configuração do Poetry;
- separação entre dependências de produção e desenvolvimento;
- versionamento do `poetry.lock`;
- configuração do Ruff;
- configuração do Pytest;
- configuração do pre-commit;
- criação do `.gitignore`;
- criação do `.dockerignore`;
- criação do `.env.example`;
- criação de settings com Pydantic Settings;
- criação de logging padronizado;
- criação de script de validação do ambiente;
- criação do Makefile;
- criação dos primeiros testes unitários e de integração;
- definição de uma seed diferente do valor convencional 42.

Principais entregas:

```text
pyproject.toml
poetry.lock
.pre-commit-config.yaml
.env.example
Makefile
src/retail_recommender/config/settings.py
src/retail_recommender/config/logging.py
scripts/validate_env.py
```

### Bloco 2 — Dataset, DVC inicial, loader e validação dos dados

Objetivo do bloco:

Incorporar o dataset RetailRocket ao projeto, versionar o dado bruto e criar uma camada confiável de carregamento e validação.

Concluído:

- instalação e inicialização do DVC;
- versionamento de `data/raw/events.csv` com DVC;
- configuração de remote local do DVC;
- criação e revisão de `configs/data.yaml`;
- criação e revisão de `params.yaml`;
- implementação do loader do RetailRocket;
- implementação do validador estrutural do `events.csv`;
- implementação do pipeline `validate_data`;
- geração do relatório de validação;
- criação do stage DVC `validate_data`;
- criação de testes unitários para o loader;
- criação de testes unitários para o validator;
- criação de teste de integração para o pipeline;
- validação com Pytest, Ruff e DVC;
- documentação das instruções de obtenção e atualização do dataset.

Principais entregas:

```text
data/raw/events.csv.dvc
src/retail_recommender/data/loaders/retailrocket_loader.py
src/retail_recommender/data/validators/events_validator.py
src/retail_recommender/pipelines/validate_data.py
artifacts/reports/data_validation.json
```

### Bloco 3 — Preprocessamento, feature engineering e preparação dos dados

Objetivo do bloco:

Transformar os eventos brutos em conjuntos processados, reproduzíveis e adequados ao treinamento e à avaliação de modelos de recomendação.

Concluído:

- implementação do padrão Strategy para preprocessadores;
- implementação do preprocessamento de feedback implícito;
- normalização das colunas do RetailRocket;
- conversão temporal dos timestamps;
- aplicação dos pesos `view`, `addtocart` e `transaction`;
- geração de `events_clean.parquet`;
- criação do stage DVC `preprocess`;
- agregação das interações por usuário e item;
- criação de `interaction_score`, `interaction_count` e `last_interaction_at`;
- filtragem de usuários e itens pouco frequentes;
- implementação do split temporal global 70/15/15;
- tratamento e contabilização de cold start;
- implementação de encoders próprios para usuários e itens;
- ajuste dos encoders exclusivamente no treino;
- persistência e carregamento dos encoders;
- implementação de negative sampling reprodutível;
- configuração de quatro negativos por positivo;
- otimização do sampling por rejeição;
- geração dos conjuntos de treino, validação e teste;
- geração do relatório de feature engineering;
- criação do stage DVC `feature_engineering`;
- integração dos três stages em um único DAG;
- criação de testes unitários e de integração;
- documentação da arquitetura e das limitações atuais.

Principais entregas:

```text
data/interim/events_clean.parquet
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet

artifacts/encoders/user_encoder.pkl
artifacts/encoders/item_encoder.pkl

artifacts/reports/feature_engineering_report.json

src/retail_recommender/data/preprocessors/
src/retail_recommender/features/
src/retail_recommender/pipelines/preprocess.py
src/retail_recommender/pipelines/feature_engineering.py
docs/architecture.md
```

## Pipeline concluído até o momento

A sequência reproduzível atual é:

```text
data/raw/events.csv
        ↓
validate_data
        ↓
artifacts/reports/data_validation.json
        ↓
preprocess
        ↓
data/interim/events_clean.parquet
        ↓
feature_engineering
        ↓
train.parquet
validation.parquet
test.parquet
user_encoder.pkl
item_encoder.pkl
feature_engineering_report.json
```

Para reproduzir toda a cadeia:

```bash
poetry run dvc repro
```

Para visualizar as dependências:

```bash
poetry run dvc dag
```

## Próximos blocos

As próximas etapas do projeto devem implementar:

1. baselines de recomendação;
2. modelo neural com PyTorch;
3. treinamento reprodutível;
4. avaliação Top-K;
5. métricas como Precision@K, Recall@K e NDCG@K;
6. comparação entre baselines e modelo neural;
7. tracking de experimentos com MLflow;
8. registro e governança do melhor modelo;
9. Model Registry;
10. integração entre DVC e MLflow;
11. empacotamento com Docker;
12. automação de qualidade e CI/CD;
13. documentação final;
14. roteiro e preparação do vídeo de apresentação.

## Conclusão

Com os três primeiros blocos concluídos, o projeto deixou de ser apenas uma estrutura inicial e passou a possuir um pipeline completo de preparação de dados para recomendação.

O Bloco 1 estabeleceu o ambiente, a organização do código e as ferramentas de qualidade. O Bloco 2 incorporou o dataset real, adicionou versionamento com DVC e criou garantias de carregamento e validação. O Bloco 3 transformou os eventos brutos em feedback implícito, construiu interações usuário-item, aplicou o split temporal, tratou entidades desconhecidas, criou encoders, gerou exemplos negativos e persistiu os conjuntos finais.

A partir deste ponto, o projeto já possui:

- dados brutos versionados;
- validação estrutural automatizada;
- preprocessamento reproduzível;
- feature engineering rastreável;
- conjuntos separados para treino, validação e teste;
- encoders persistidos;
- negative sampling reprodutível;
- três stages DVC integrados;
- testes automatizados;
- documentação das decisões arquiteturais.

O próximo passo natural é utilizar esses artefatos para construir, treinar e comparar os modelos de recomendação.
