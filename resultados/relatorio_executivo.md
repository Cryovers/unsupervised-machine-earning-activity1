# Relatório executivo — Segmentação de clientes

## 1. Sumário executivo

- A base contém 541,909 linhas. Após a limpeza, 392,692 linhas de compras válidas representam 4,338 clientes e £8,887,208.89 de receita bruta observada.
- O modelo selecionado foi **Hierarquico_average_2**, com 2 segmentos. A maior concentração de receita está em **Valiosos em risco (C1)**, responsável por 52.5% do total.
- O segmento **Valiosos em risco (C1)** apresenta a maior recência média: 98 dias sem compra. Isso sinaliza inatividade relativa, sem comprovar abandono definitivo.

Três prioridades: (1) proteger clientes de maior valor com fidelização e atendimento direcionado; (2) testar reativação dos segmentos inativos; (3) estimular uma nova compra nos segmentos de menor frequência. Validar todas as intervenções com grupos de controle antes de ampliar investimentos.

## 2. Metodologia e validação

Foi utilizada a base UCI Online Retail, com vendas de dezembro de 2010 a dezembro de 2011, em libras esterlinas. Foram removidas duplicatas exatas, linhas sem cliente, campos essenciais inválidos, cancelamentos e quantidades/preços não positivos, nessa ordem. A auditoria registra as exclusões sem dupla contagem. A receita mede vendas positivas identificadas, não receita líquida de devoluções, margem ou lucro.

Cada cliente é representado por recência em dias até 2011-12-10, frequência de notas fiscais distintas e soma monetária. Foram aplicados log1p e StandardScaler para reduzir assimetria e equilibrar escalas, preservando grandes compradores. Ticket médio, diversidade de produtos e tempo observado complementam a interpretação, mas não entram no agrupamento.

A busca compara K-Means e hierárquico ward/complete/average com k entre 2 e 8; GMM full/diag com 1 a 8 componentes, cinco inicializações e regularização. Candidatos operacionais exigem convergência e no mínimo 2% dos clientes em cada grupo. Dentro de cada abordagem, foi utilizado o maior silhouette, ou o menor BIC para GMM. Entre candidatos, vence a menor média dos rankings de silhouette, Davies–Bouldin e Calinski–Harabasz, com desempate por silhouette. É uma regra explícita de compromisso, não um ótimo universal.

O vencedor obteve silhouette **0.466**, Davies–Bouldin **0.656** e Calinski–Harabasz **1459.6**. Métricas usam todos os clientes no mesmo espaço RFM transformado. Tempos medem ajuste e atribuição, excluindo cálculo de métricas. Não há rótulos verdadeiros; validação interna não demonstra retorno financeiro.

## 3. Resultados e insights

- **Campeões e fiéis (C0)**: 247 clientes (5.7%), 47.5% da receita; recência média 8 dias, 24.0 pedidos e £17,104 por cliente. Benefícios de fidelidade e venda complementar; medir retenção e margem em 90 dias.
- **Valiosos em risco (C1)**: 4091 clientes (94.3%), 52.5% da receita; recência média 98 dias, 3.1 pedidos e £1,140 por cliente. Contato personalizado de reativação; medir recompra incremental em 60 dias.

O cotovelo geométrico do K-Means sugere k=4; a escolha por silhouette e tamanho mínimo foi KMeans_2. Discordância evidencia que compactação e separação são critérios diferentes. Dez sementes com 20 inicializações produziram ARI médio entre pares de 1.000; as iterações e inércias estão no arquivo de estabilidade. Estabilidade entre sementes não garante estabilidade ao longo do tempo.

No hierárquico, a configuração operacional com melhor silhouette usa average, k=2, com corte a 3.61 e salto de ligação de 0.73. A linha vermelha separa os ramos; o desenho exibe os últimos 30 ramos de uma árvore calculada com todos os clientes. Complete/average podem isolar grandes compradores, razão para explicitar a restrição de tamanho.

O maior segmento contém 94.3% dos clientes: essa concentração reduz a granularidade comercial e pode ocultar subgrupos. Avaliar k=3 a 8 em testes de campanhas é um próximo passo, mesmo que as métricas internas sejam menores.

O menor BIC global foi de GMM_full_8; o menor AIC foi de GMM_full_8. Esses mínimos são reportados mesmo quando não atendem à regra operacional. No GMM de menor BIC, 272 clientes têm probabilidade máxima inferior a 70%: casos ambíguos que merecem campanhas menos específicas. Probabilidades são pertinências do modelo, não probabilidades de compra. Um mínimo em oito componentes está na fronteira da busca e não comprova ótimo global.

PCA em duas dimensões retém 93.9% da variância; 2 componentes atingem 80%, e a visualização 3D preserva 100%. PCA é usado apenas para visualizar. As cargas assinadas estão no notebook: sua magnitude mostra quais variáveis influenciam cada eixo; o sinal dos eixos é arbitrário. Centroides são médias dos grupos projetadas, inclusive quando o algoritmo não possui centroides próprios.

## 4. Recomendações, limitações e próximos passos

Marketing/CRM deve conduzir testes aleatorizados por segmento, com controle sem campanha, comparando recompra, receita por cliente e margem incremental em 30, 60 e 90 dias. Finanças deve definir um teto de custo por cliente a partir da margem disponível. Não é possível estimar ROI sem custos, descontos e efeito causal. Priorizar segmentos combinando tamanho, valor e recência; evitar oferecer descontos generalizados aos compradores já fiéis.

Clientes recentes não são necessariamente novos: a janela limita a observação. Os nomes são regras descritivas baseadas nas médias dos segmentos e medianas globais, não classes RFM 555 nem diagnósticos individuais. A empresa inclui atacadistas; valores extremos podem ser legítimos. Cancelamentos excluídos, compradores anônimos, sazonalidade e dados históricos limitam a generalização. Duplicatas exatas podem representar compras legítimas; a decisão deve ser auditada com a origem. Como evolução, comparar janelas temporais, incluir devoluções líquidas, avaliar estabilidade por reamostragem e revisar segmentos mensalmente.

Fonte: Chen, D. (2015). [Online Retail, UCI](https://doi.org/10.24432/C5BW33), CC BY 4.0.
