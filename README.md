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

## Escopo atual do projeto

O projeto está organizado em blocos incrementais.

### Bloco 1 — Estrutura inicial, Clean Code e ambiente base

O Bloco 1 configura a fundação técnica do projeto.

Incluído no Bloco 1:

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

O Bloco 2 prepara a base RetailRocket para uso no projeto.

Incluído no Bloco 2:

- posicionamento do arquivo bruto `events.csv` em `data/raw/`;
- inicialização do DVC;
- versionamento do arquivo bruto com DVC;
- configuração inicial de parâmetros de dados;
- criação do loader do RetailRocket;
- criação do validador estrutural do dataset;
- criação do pipeline `validate_data`;
- geração do relatório `artifacts/reports/data_validation.json`;
- criação do primeiro stage DVC em `dvc.yaml`;
- testes unitários e de integração relacionados ao loader, ao validador e ao pipeline.

Fora do escopo do Bloco 2:

- preprocessamento completo;
- Strategy Pattern de preprocessamento;
- agregação usuário-item;
- split temporal;
- encoders;
- negative sampling;
- baselines;
- modelo neural PyTorch;
- métricas de ranking;
- MLflow;
- Docker;
- Model Registry.

Esses itens serão implementados nos próximos blocos.

## Estrutura inicial do projeto

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
│   └── integration/
├── scripts/
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── poetry.lock
├── .env.example
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── README.md
└── Makefile
```

Observação: o arquivo real `data/raw/events.csv` existe localmente, mas não deve ser versionado diretamente pelo Git. O Git deve versionar apenas o arquivo de metadados `data/raw/events.csv.dvc`.

## Principais diretórios

### `src/retail_recommender/`

Contém o código-fonte principal da aplicação.

### `src/retail_recommender/data/loaders/`

Contém componentes responsáveis pela leitura de dados brutos.

No Bloco 2, o principal loader será:

```text
src/retail_recommender/data/loaders/retailrocket_loader.py
```

Esse módulo deve carregar o arquivo `events.csv` e retornar uma estrutura de dados adequada para validação e etapas futuras.

### `src/retail_recommender/data/validators/`

Contém validadores estruturais dos dados.

No Bloco 2, o principal validador será:

```text
src/retail_recommender/data/validators/events_validator.py
```

Esse módulo deve verificar se o dataset possui o contrato mínimo esperado para o problema de recomendação.

### `src/retail_recommender/pipelines/`

Contém pipelines executáveis do projeto.

No Bloco 2, será criado o pipeline:

```text
src/retail_recommender/pipelines/validate_data.py
```

Esse pipeline deve carregar o dataset, executar a validação estrutural e salvar um relatório em JSON.

### `configs/`

Contém arquivos de configuração em YAML.

No Bloco 2, o arquivo principal é:

```text
configs/data.yaml
```

Ele concentra caminhos e regras básicas do dataset, como colunas obrigatórias, eventos permitidos e limites mínimos de interações, usuários e itens.

### `data/`

Contém os dados locais do projeto.

A pasta é dividida em:

- `data/raw/`: dados brutos;
- `data/interim/`: dados intermediários;
- `data/processed/`: dados prontos para modelagem.

Os dados reais não devem ser versionados diretamente no Git. O versionamento dos dados é feito com DVC.

### `artifacts/`

Contém artefatos gerados pelo projeto, como modelos treinados, encoders e relatórios.

No Bloco 2, a validação de dados deve gerar:

```text
artifacts/reports/data_validation.json
```

### `tests/`

Contém testes unitários e de integração.

No Bloco 2, devem ser adicionados testes para:

- loader do RetailRocket;
- validador de eventos;
- pipeline de validação de dados com fixture pequena.

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

## Dataset

O dataset escolhido é o RetailRocket.

Neste projeto, o principal arquivo utilizado será:

```text
events.csv
```

Ele contém eventos de interação entre usuários e produtos, como:

- `view`;
- `addtocart`;
- `transaction`.

O arquivo deve ser colocado localmente em:

```text
data/raw/events.csv
```

O arquivo `events.csv` não deve ser commitado diretamente no Git.

O versionamento do dataset bruto é feito com DVC. O Git deve versionar apenas o arquivo `.dvc` correspondente:

```text
data/raw/events.csv.dvc
```

## Contrato esperado do `events.csv`

O arquivo `events.csv` deve conter, no mínimo, as seguintes colunas:

- `timestamp`;
- `visitorid`;
- `event`;
- `itemid`;
- `transactionid`.

A validação estrutural deve verificar:

- se as colunas obrigatórias existem;
- se `timestamp`, `visitorid`, `event` e `itemid` não estão totalmente vazios;
- se os eventos pertencem ao conjunto esperado: `view`, `addtocart`, `transaction`;
- se há pelo menos 10.000 interações no dataset real;
- se é possível converter `timestamp` para datetime;
- se existem usuários e itens suficientes para um problema de recomendação;
- se o relatório de validação é salvo em JSON.

## DVC

O projeto usa DVC para versionamento de dados e, nos próximos blocos, para definição de pipelines reprodutíveis.

### Inicialização do DVC

Para inicializar o DVC no projeto:

```bash
poetry run dvc init
```

### Adição do dataset bruto

Após colocar o arquivo `events.csv` em `data/raw/`, versione o arquivo com:

```bash
poetry run dvc add data/raw/events.csv
```

Esse comando cria o arquivo:

```text
data/raw/events.csv.dvc
```

Depois, envie os dados para o remote configurado:

```bash
poetry run dvc push
```

### Recuperação dos dados

Após clonar o projeto em outra máquina, instale as dependências e recupere os dados com:

```bash
poetry install
poetry run dvc pull
```

### Verificação do status do DVC

Para verificar se os dados e pipelines estão atualizados:

```bash
poetry run dvc status
```

Resultado esperado quando tudo estiver correto:

```text
Data and pipelines are up to date.
```

## Configurações de dados

O arquivo `configs/data.yaml` concentra configurações específicas do dataset.

Exemplo esperado:

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

## Parâmetros do projeto

O arquivo `params.yaml` concentra parâmetros rastreáveis pelo DVC.

Exemplo inicial:

```yaml
data:
  raw_events_path: data/raw/events.csv
  validation_report_path: artifacts/reports/data_validation.json
  minimum_interactions: 10000
  minimum_users: 100
  minimum_items: 100

events:
  weights:
    view: 1
    addtocart: 3
    transaction: 5
```

## Pipeline de validação de dados

O primeiro pipeline do projeto será o `validate_data`.

Objetivo:

1. carregar `data/raw/events.csv`;
2. validar a estrutura mínima do dataset;
3. gerar o relatório `artifacts/reports/data_validation.json`.

O stage correspondente será registrado no `dvc.yaml`.

Exemplo de execução:

```bash
poetry run dvc repro validate_data
```

Após a execução, o relatório esperado é:

```text
artifacts/reports/data_validation.json
```

## Testes

Para rodar os testes:

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

## Estratégia inicial de modelagem

A formulação será baseada em recomendação com feedback implícito.

A princípio, os eventos receberão pesos diferentes:

```text
view = 1
addtocart = 3
transaction = 5
```

Esses pesos poderão ser ajustados em experimentos futuros e rastreados com MLflow.

A implementação de modelos, baselines, treino, métricas de ranking e tracking de experimentos está fora do escopo do Bloco 2.

## Reprodutibilidade

O projeto usará seeds fixas para reduzir variação entre execuções.

A seed inicial configurada é:

```text
RANDOM_SEED=317
```

Nos blocos futuros, essa seed será aplicada em:

- Python;
- NumPy;
- Scikit-Learn;
- PyTorch.

Além disso, o projeto usará DVC para garantir reprodutibilidade de dados e pipelines.

## Fluxo de branches e commits

Cada funcionalidade nova deve ser desenvolvida em uma branch específica, acompanhada de testes ou validação por comando.

Exemplos de branches para o Bloco 2:

```text
feature/dvc-initial-setup
feature/retailrocket-loader
feature/events-validator
feature/validate-data-pipeline
docs/update-dataset-instructions
```

Exemplos de commits semânticos:

```text
chore: initialize dvc data versioning
feat: add retailrocket events loader
feat: add retailrocket events validation
feat: add data validation dvc stage
docs: update dataset and dvc instructions
```

Antes de cada commit, rode:

```bash
poetry run pytest
poetry run ruff check .
```

Quando houver alterações de DVC, rode também:

```bash
poetry run dvc status
```

Para o stage de validação de dados, rode:

```bash
poetry run dvc repro validate_data
```

## Status atual

Bloco atual:

```text
Bloco 2 — Dataset, DVC inicial, loader e validação dos dados
```

Status:

```text
Em desenvolvimento
```

## Próximos blocos

Os próximos blocos devem implementar:

1. preprocessamento de feedback implícito;
2. agregação usuário-item;
3. engenharia de features;
4. split temporal;
5. encoders;
6. amostragem negativa;
7. baselines de recomendação;
8. modelo neural com PyTorch;
9. avaliação Top-K;
10. tracking com MLflow;
11. Model Registry;
12. Dockerfile multi-stage;
13. docker-compose;
14. documentação final;
15. roteiro do vídeo.
