# Model Card — RetailRocket Item-KNN Recommender

## 1. Identificação

**Nome do sistema:** RetailRocket Recommender
**Modelo selecionado:** Item-KNN
**Modelo registrado:** `RetailRocketRecommender`
**Alias operacional:** `staging`
**Tipo:** recomendador Top-K com feedback implícito
**Cutoff de avaliação:** K = 10

## 2. Objetivo

O modelo recomenda produtos para usuários conhecidos com base no histórico de interações observadas no e-commerce.

O objetivo é gerar uma lista Top-10 de itens com maior relevância estimada para cada usuário.

## 3. Dataset

O projeto utiliza o dataset RetailRocket, principalmente o arquivo:

```text
events.csv
```

Eventos considerados:

| Evento | Peso |
| --- | ---: |
| `view` | 1.0 |
| `addtocart` | 3.0 |
| `transaction` | 5.0 |

Os pesos representam a intensidade relativa do sinal de interesse e são configurados em `params.yaml`.

## 4. Tipo de feedback

O dataset possui feedback implícito.

Não existem notas explícitas. Uma interação indica interesse observado, mas a ausência de interação não representa rejeição.

## 5. Processamento

O pipeline:

1. valida o arquivo bruto;
2. remove registros inválidos;
3. converte timestamps;
4. agrega eventos por usuário e item;
5. calcula score e frequência;
6. executa split temporal;
7. ajusta encoders no treino;
8. remove entidades desconhecidas;
9. gera datasets para baselines e modelo neural.

## 6. Split

O split é temporal:

```text
70% treino
15% validação
15% teste
```

As interações são ordenadas por `last_interaction_at`.

Essa abordagem aproxima a avaliação de um cenário no qual o modelo aprende com o passado e prevê interações futuras.

## 7. Modelos comparados

- Popularity;
- Item-KNN;
- Neural Collaborative Filtering.

## 8. Métricas

- Precision@10;
- Recall@10;
- NDCG@10;
- MAP@10;
- Coverage@10.

A métrica primária de seleção é NDCG@10.

## 9. Resultado atual

O Item-KNN apresentou o melhor NDCG@10 na configuração avaliada.

| Métrica | Resultado |
| --- | ---: |
| NDCG@10 | 0.040 |
| Recall@10 | 0.050 |
| MAP@10 | 0.034 |
| Coverage@10 | 0.00546 |
| Usuários avaliados | 50 |

Os valores devem ser interpretados como resultados de desenvolvimento, pois a avaliação utiliza uma amostra limitada de usuários.

## 10. Decisão

O Item-KNN foi selecionado porque apresentou a melhor métrica primária entre os modelos comparados.

O modelo foi registrado no MLflow Model Registry e recebeu o alias:

```text
staging
```

A promoção para produção não foi automatizada.

## 11. Entradas

O modelo recebe um DataFrame contendo:

```text
user_idx
```

O usuário precisa estar presente no encoder utilizado no treinamento.

## 12. Saída

A saída contém:

```text
user_idx
recommendations
```

`recommendations` representa a lista ordenada de itens sugeridos.

## 13. Uso recomendado

- experimentação de recomendação Top-K;
- comparação de estratégias;
- demonstração de pipeline MLOps;
- recomendação para usuários conhecidos;
- validação offline de arquitetura.

## 14. Uso não recomendado

O modelo não deve ser utilizado diretamente para:

- decisões financeiras;
- decisões de crédito;
- contextos médicos;
- personalização para menores sem análise adicional;
- serving de produção sem testes de latência;
- usuários ou itens totalmente novos;
- decisões com impacto jurídico ou regulatório.

## 15. Limitações

- cobertura baixa;
- dependência de usuários e itens conhecidos;
- viés em favor de itens com histórico;
- ausência de features de conteúdo;
- avaliação limitada a 50 usuários;
- ausência de tuning extensivo;
- comportamento sensível à janela temporal;
- ausência de métricas online;
- feedback implícito com pseudo-negativos.

## 16. Cold start

Usuários e itens desconhecidos no treino recebem inicialmente índice `-1` e são removidos da avaliação quando o filtro correspondente está habilitado.

O modelo não possui estratégia específica para cold start.

Uma solução futura pode combinar:

- itens populares;
- categorias;
- propriedades dos itens;
- contexto de sessão;
- modelo híbrido.

## 17. Riscos e vieses

O histórico observado pode refletir:

- exposição desigual dos itens;
- campanhas promocionais;
- posição dos produtos na interface;
- disponibilidade de estoque;
- sazonalidade;
- comportamento de usuários mais ativos.

O modelo pode reforçar itens já expostos e reduzir diversidade.

## 18. Falsos negativos

Itens não observados são tratados como desconhecidos, não como rejeições explícitas.

No treino neural, parte desses pares é amostrada como pseudo-negativa. Alguns deles podem representar produtos relevantes ainda não exibidos ao usuário.

## 19. Overfitting

O Neural CF utiliza early stopping baseado na loss de validação.

O Item-KNN não possui treinamento iterativo, mas pode se ajustar demais a padrões locais do histórico quando o catálogo é esparso ou quando o número de vizinhos não é adequado.

## 20. Reprodutibilidade

A execução depende de:

- `poetry.lock`;
- `params.yaml`;
- arquivos de configuração;
- `dvc.yaml`;
- `dvc.lock`;
- dados versionados pelo DVC;
- random seed centralizada;
- imagem Docker.

## 21. Registro e governança

O modelo é registrado como:

```text
RetailRocketRecommender
```

A versão validada recebe o alias:

```text
staging
```

A promoção para `production` deve exigir critérios adicionais, como:

- avaliação com amostra representativa;
- validação de negócio;
- testes de latência;
- testes de estabilidade;
- verificação de cobertura;
- plano de rollback.

## 22. Próximos passos

- ampliar a avaliação;
- avaliar diferentes valores de vizinhos;
- medir diversidade e novidade;
- adicionar estratégia de cold start;
- avaliar modelo híbrido;
- criar testes de latência;
- estabelecer critérios formais para produção;
- monitorar drift e cobertura.
