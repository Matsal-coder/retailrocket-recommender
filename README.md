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

## Escopo do Bloco 1

Este bloco configura apenas a fundação técnica do projeto.

Incluído neste bloco:

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

Fora do escopo deste bloco:

- DVC;
- MLflow;
- Dockerfile;
- docker-compose;
- modelos de recomendação;
- baselines;
- treino com PyTorch;
- processamento real do dataset RetailRocket;
- métricas de ranking;
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
│       │   ├── validators/
│       │   └── preprocessors/
│       ├── features/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── tracking/
│       └── pipelines/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── artifacts/
│   ├── models/
│   ├── encoders/
│   └── reports/
├── docs/
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
├── pyproject.toml
├── poetry.lock
├── .env.example
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── README.md
└── Makefile
```

## Principais diretórios

### `src/retail_recommender/`

Contém o código-fonte principal da aplicação.

### `configs/`

Contém arquivos de configuração em YAML. Nos próximos blocos, esses arquivos serão usados para parametrizar dados, modelos, treino e avaliação.

### `data/`

Contém os dados locais do projeto.

A pasta é dividida em:

- `data/raw/`: dados brutos;
- `data/interim/`: dados intermediários;
- `data/processed/`: dados prontos para modelagem.

Os dados reais não devem ser versionados diretamente no Git.

### `artifacts/`

Contém artefatos gerados pelo projeto, como modelos treinados, encoders e relatórios.

### `tests/`

Contém testes unitários e de integração.

### `scripts/`

Contém scripts auxiliares de desenvolvimento e operação.

## Requisitos

- Python >= 3.11 e < 3.13
- Poetry
- Git

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
RANDOM_SEED=42
```

O arquivo `.env.example` deve ser versionado.

O arquivo `.env` local não deve ser versionado.

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

A ingestão e validação real do dataset será implementada em bloco futuro.

## Estratégia inicial de modelagem

A formulação será baseada em recomendação com feedback implícito.

A princípio, os eventos receberão pesos diferentes:

```text
view = 1
addtocart = 3
transaction = 5
```

Esses pesos poderão ser ajustados em experimentos futuros e rastreados com MLflow.

## Reprodutibilidade

O projeto usará seeds fixas para reduzir variação entre execuções.

A seed inicial configurada é:

```text
RANDOM_SEED=42
```

Nos blocos futuros, essa seed será aplicada em:

- Python;
- NumPy;
- Scikit-Learn;
- PyTorch.

## Status atual

Bloco atual:

```text
Bloco 1 — Estrutura inicial, Clean Code e ambiente base
```

Status:

```text
Em desenvolvimento
```

## Próximos blocos

Os próximos blocos devem implementar:

1. ingestão e validação do dataset RetailRocket;
2. pré-processamento de feedback implícito;
3. engenharia de features;
4. split temporal;
5. amostragem negativa;
6. baselines de recomendação;
7. modelo neural com PyTorch;
8. avaliação Top-K;
9. tracking com MLflow;
10. pipeline reprodutível com DVC;
11. empacotamento com Docker;
12. documentação final e roteiro do vídeo.
