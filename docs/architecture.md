# Arquitetura do pipeline de recomendação

## Visão geral

O sistema utiliza feedback implícito do dataset RetailRocket para construir conjuntos de dados preparados para um modelo de recomendação Top-K.

O fluxo atual é:

events.csv
    ↓
validação
    ↓
events_clean.parquet
    ↓
agregação usuário-item
    ↓
split temporal
    ↓
encoders
    ↓
negative sampling
    ↓
train / validation / test

## Eventos e pesos

Os eventos suportados são:

- view: 1.0
- addtocart: 3.0
- transaction: 5.0

A pontuação agregada de uma interação usuário-item é a soma dos pesos dos eventos observados.

## Construção das interações

Os eventos são agregados por:

user_id
item_id

O resultado contém:

interaction_score
interaction_count
last_interaction_at
target

O campo target é inicialmente igual a 1, porque representa uma interação positiva observada.

## Split temporal

As interações são ordenadas por last_interaction_at.

O corte utilizado é:

70% treino
15% validação
15% teste

A divisão global no tempo aproxima o cenário em que o modelo é treinado com o histórico disponível e avaliado no futuro.

Essa escolha reduz vazamento temporal, mas pode expor sazonalidade e mudanças de distribuição entre os períodos. Por isso, os intervalos de datas dos conjuntos são registrados no relatório de feature engineering.

## Cold start

Somente usuários e itens presentes no treino são considerados na avaliação inicial.

Interações desconhecidas em validação e teste são removidas e contabilizadas no relatório.

A resolução completa de cold start está fora do escopo da versão atual.

## Encoders

Os IDs originais são convertidos em índices contínuos.

Os encoders são ajustados somente no treino e persistidos para garantir consistência entre treinamento, avaliação e inferência.

IDs desconhecidos recebem temporariamente o índice -1, mas não são mantidos nos conjuntos finais.

## Positive e negative sampling

As interações observadas são tratadas como exemplos positivos:

target = 1

No pipeline atual, todos os pares positivos do conjunto de treino são preservados.

Como o dataset possui feedback implícito, a ausência de interação não representa rejeição explícita. Ainda assim, pares usuário-item não observados podem ser usados como pseudo-negativos:

target = 0

O conjunto de treino combina:

positivos observados
+
negativos amostrados

A configuração inicial utiliza quatro negativos por positivo.

Essa razão é um hiperparâmetro e foi escolhida como baseline para oferecer contraste suficiente ao modelo sem tornar o conjunto de treino excessivamente grande.

## Estratégia de negative sampling

O algoritmo gera negativos somente com usuários e itens conhecidos no treino.

Para cada usuário:

1. identifica os itens positivos conhecidos;
2. sorteia candidatos no intervalo dos itens codificados;
3. rejeita candidatos que já sejam positivos;
4. continua até gerar a quantidade configurada de negativos.

A implementação usa amostragem por rejeição para evitar percorrer o catálogo inteiro para cada usuário.

## Outputs

O pipeline produz:

data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet

artifacts/encoders/user_encoder.pkl
artifacts/encoders/item_encoder.pkl

artifacts/reports/feature_engineering_report.json

## DVC

O pipeline reprodutível contém três stages:

validate_data
      ↓
preprocess
      ↓
feature_engineering

O stage validate_data valida o dataset bruto.

O stage preprocess transforma os eventos em feedback implícito padronizado.

O stage feature_engineering constrói as interações, realiza o split temporal, ajusta os encoders, gera negativos e persiste os conjuntos finais.

## Decisões atuais

As principais decisões arquiteturais são:

- feedback implícito;
- pesos diferentes por evento;
- agregação usuário-item;
- split temporal global;
- encoders ajustados somente no treino;
- remoção de desconhecidos na avaliação;
- negative sampling aleatório;
- quatro negativos por positivo;
- amostragem por rejeição;
- DVC para versionamento e reprodução dos artefatos.

## Limitações

A versão atual não resolve:

- cold start de novos usuários;
- cold start de novos itens;
- sazonalidade explicitamente;
- hard negative sampling;
- features de conteúdo dos produtos;
- ranking contra todo o catálogo;
- retreinamento por janela móvel;
- avaliação walk-forward;
- possível presença de falsos negativos entre pares não observados.

Esses pontos poderão ser tratados em blocos futuros.
