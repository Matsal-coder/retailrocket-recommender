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

## Pipeline de validação dos dados

O pipeline de validação está implementado em:

```text
src/retail_recommender/pipelines/validate_data.py
```

Para rodar manualmente:

```bash
poetry run python -m retail_recommender.pipelines.validate_data
```

O pipeline:

1. lê as configurações de `configs/data.yaml`;
2. carrega `data/raw/events.csv`;
3. valida a estrutura do dataset;
4. salva o relatório em `artifacts/reports/data_validation.json`;
5. falha com erro claro caso a validação não seja aprovada.

## Stage DVC `validate_data`

O primeiro stage do DVC é:

```text
validate_data
```

Ele está definido em:

```text
dvc.yaml
```

Para executar o stage:

```bash
poetry run dvc repro validate_data
```

Para verificar se o pipeline está atualizado:

```bash
poetry run dvc status
```

## Relatório de validação

O relatório de validação é salvo em:

```text
artifacts/reports/data_validation.json
```

Exemplo de estrutura:

```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [],
  "summary": {
    "rows": 2756101,
    "columns": [
      "timestamp",
      "visitorid",
      "event",
      "itemid",
      "transactionid"
    ],
    "unique_users": 1407580,
    "unique_items": 235061,
    "event_counts": {
      "view": 2664312,
      "addtocart": 69332,
      "transaction": 22457
    },
    "min_timestamp": "2015-05-03T03:00:04.384000",
    "max_timestamp": "2015-09-18T02:59:47.788000"
  }
}
```

Os números acima são ilustrativos e podem variar conforme o arquivo utilizado.

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
poetry run dvc repro validate_data
poetry run dvc status
```

Resultado esperado:

- testes passando;
- Ruff sem erros;
- stage `validate_data` reproduzível;
- DVC sem pendências.

## Estratégia inicial de modelagem

A formulação será baseada em recomendação com feedback implícito.

A princípio, os eventos receberão pesos diferentes:

```text
view = 1
addtocart = 3
transaction = 5
```

Esses pesos poderão ser ajustados em experimentos futuros e rastreados com MLflow.

A aplicação efetiva desses pesos ainda não foi implementada neste bloco.

## Reprodutibilidade

O projeto usa seeds fixas para reduzir variação entre execuções.

A seed inicial configurada é:

```text
RANDOM_SEED=317
```

Nos próximos blocos, essa seed será aplicada em:

- Python;
- NumPy;
- Scikit-Learn;
- PyTorch.

## Status atual

Bloco atual concluído:

```text
Bloco 2 — Dataset, DVC inicial, loader e validação dos dados
```

Status:

```text
Concluído
```

## O que foi concluído até aqui

### Bloco 1 — Estrutura inicial, Clean Code e ambiente base

Concluído:

- criação da estrutura inicial de pastas;
- configuração do Poetry;
- separação entre dependências de produção e desenvolvimento;
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
- criação dos primeiros testes básicos.

### Bloco 2 — Dataset, DVC inicial, loader e validação dos dados

Concluído:

- criação da branch de setup inicial do DVC;
- instalação e inicialização do DVC;
- versionamento de `data/raw/events.csv` com DVC;
- configuração de remote local do DVC;
- criação/revisão de `configs/data.yaml`;
- criação/revisão de `params.yaml`;
- implementação do loader do RetailRocket;
- implementação do validador estrutural do `events.csv`;
- implementação do pipeline `validate_data`;
- geração do relatório `artifacts/reports/data_validation.json`;
- criação do stage DVC `validate_data`;
- criação de testes unitários para loader;
- criação de testes unitários para validator;
- criação de teste de integração para o pipeline;
- validação com Pytest;
- validação com Ruff;
- validação com `dvc repro validate_data`.

## Próximos blocos

Os próximos blocos devem implementar:

1. preprocessamento de feedback implícito;
2. aplicação dos pesos por tipo de evento;
3. agregação usuário-item;
4. engenharia de features;
5. split temporal;
6. encoders de usuários e itens;
7. amostragem negativa;
8. baselines de recomendação;
9. modelo neural com PyTorch;
10. avaliação Top-K;
11. tracking com MLflow;
12. Model Registry;
13. pipeline DVC completo;
14. empacotamento com Docker;
15. documentação final e roteiro do vídeo.
