# Atividade 1 — Segmentação de clientes

Implementação do enunciado `ATIVIDADE1 (1).pdf`, usando **UCI Online Retail** e dados reais. O PDF anuncia cinco itens, mas detalha três; todos os requisitos detalhados estão mapeados abaixo.

## Entregáveis

- `Atividade1_jonatan_,goulart_maciel.ipynb`: notebook documentado e executável; as saídas são geradas ao executar as células.
- `atividade1.py`: pipeline completo, reproduzível, com download automático.
- `resultados/dashboard.html`: dashboard Plotly interativo gerado localmente por `python atividade1.py`. Após a geração, funciona offline. O HTML e as saídas gráficas individuais não estão versionados nesta publicação.
- `resultados/relatorio_executivo.md`: relatório com até 1.500 palavras e ações de negócio.
- `resultados/*.csv`: comparações, seleção, perfis agregados, estabilidade e cargas PCA. Os arquivos individuais `clientes_segmentados.csv` e `probabilidades_gmm.csv` são gerados localmente pelo pipeline e não são versionados no repositório público.
- `resultados/auditoria.json` e `resumo.json`: exclusões, escolhas, semente e SHA-256 dos dados.

## Execução

Python 3.11 ou superior. Na pasta do projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python atividade1.py
```

No Windows, ative com `.venv\Scripts\activate`. Se `venv` não estiver disponível, instale o componente de ambiente virtual da sua distribuição Python. A primeira execução baixa cerca de 23 MB da UCI; execuções posteriores reutilizam `dados/Online Retail.xlsx`. O ajuste utiliza todos os clientes, sem amostragem. Reserve alguns minutos e cerca de 1 GB de memória, pois as métricas e o hierárquico usam distâncias entre pares. Os tempos variam por máquina.

Para usar um arquivo já disponível:

```bash
python atividade1.py --dados "dados/Online Retail.xlsx" --saida resultados
```

CSV também é aceito, com colunas do esquema UCI. A busca foi projetada para essa base, não para datasets arbitrários. Abra o notebook no Jupyter ou VS Code, selecione o ambiente com as dependências e execute as células na ordem. A célula do pipeline regenera todos os resultados; as seguintes explicam e exibem os artefatos. Os arquivos `requirements.txt` permitem versões compatíveis; `requirements-lock.txt` registra as versões exatas do ambiente utilizado.

## Decisões metodológicas

- Recency: dias desde a última compra, até o dia seguinte à data máxima da base. Frequency: **notas fiscais únicas**, não linhas de produtos. Monetary: soma de quantidade × preço em GBP.
- Exclusões sequenciais: duplicatas exatas, cliente ausente, campos inválidos, cancelamentos, quantidade não positiva, preço não positivo. Descrição ausente não impede o uso de uma venda válida.
- Log1p e StandardScaler sobre RFM; grandes compradores foram preservados. Variáveis extras (ticket, produtos distintos, tempo observado) são descritivas.
- K-Means: k=2…8, 20 inicializações, até 500 iterações; cotovelo geométrico e silhouette; dez sementes para convergência e estabilidade via ARI.
- Hierárquico: ward, complete e average, k=2…8. Dendrograma calculado com todos os clientes; truncamento só na exibição. Corte por silhouette com restrição de tamanho, acompanhado da altura e salto de ligação.
- GMM (alternativa ao DBSCAN autorizada no PDF): 1…8 componentes, covariâncias full/diag, cinco inicializações, BIC/AIC, probabilidades individuais e indicação de pertinência incerta (<70%).
- Segmentos operacionais devem conter pelo menos 2% dos clientes. A regra evita escolher partições que isolam poucos extremos; não implica que esses clientes sejam irrelevantes. Todos os resultados, inclusive inelegíveis, permanecem na comparação.
- Seleção dentro de cada abordagem por silhouette ou BIC; comparação dos candidatos pela média de rankings de silhouette (maior), Davies–Bouldin (menor) e Calinski–Harabasz (maior). Tempo é informativo, não critério de seleção. Sem rótulos externos ou validação temporal, os resultados são exploratórios.
- PCA não altera o ajuste. São informados a variância acumulada e o número de componentes necessário para atingir 80%; se a projeção 2D não atingir isso em outra base, a projeção 3D retém as três variáveis. Centros mostrados são médias projetadas de cada grupo.
- Radar usa médias padronizadas reescaladas entre segmentos; recência maior significa **menos** atividade recente. Heatmap usa cor padronizada e médias originais nas anotações.
- Nomes dos segmentos seguem regras descritivas explícitas, comparando médias RFM com medianas da base. Não se deve interpretar clientes recentes como comprovadamente novos nem inatividade como churn confirmado.

## Correspondência com o PDF

| Requisito | Implementação / evidência |
|---|---|
| Três algoritmos e otimização | `comparar`, `comparacao_modelos.csv`, `selecao_modelos.csv` |
| Cotovelo, silhouette, convergência e estabilidade | dashboard e `estabilidade_kmeans.csv` |
| Linkages, dendrograma e corte | dashboard, `dendrograma.png`, `resumo.json` |
| GMM, BIC/AIC e probabilidades | dashboard e `probabilidades_gmm.csv` |
| Silhouette, Davies–Bouldin, Calinski–Harabasz, inércia e tempo | `comparacao_modelos.csv` |
| PCA 2D/3D, variância, centroides e componentes | dashboard, notebook e `pca_cargas.csv` |
| Radar e heatmap normalizado com médias | dashboard e notebook |
| Barras, boxplot e pizza | dashboard e notebook |
| Segmentos nomeados e oportunidades | `perfis_segmentos.csv` e relatório |
| Relatório executivo e top 3 recomendações | `relatorio_executivo.md` |
| Dashboard interativo e notebook reproduzível | `dashboard.html`, `Atividade1_jonatan_,goulart_maciel.ipynb` |
| Código versionado no GitHub | [Repositório público](https://github.com/Cryovers/unsupervised-machine-earning-activity1) |

O código e os entregáveis estão versionados no [GitHub](https://github.com/Cryovers/unsupervised-machine-earning-activity1). Os dados brutos são ignorados e podem ser baixados novamente pelo pipeline. Para visualizar o dashboard, baixe `resultados/dashboard.html` e abra o arquivo no navegador.

## Verificação

```bash
python -m unittest discover -s tests -v
```

Os testes cobrem contagem de pedidos, recência, auditoria das exclusões, conservação da receita e rejeição de entradas inválidas. A análise também foi executada na base real; a cópia local contém as saídas salvas; a versão pública permite regenerá-las.

## Referências

- Chen, D. (2015). [Online Retail — UCI](https://archive.ics.uci.edu/dataset/352/online+retail). DOI: https://doi.org/10.24432/C5BW33. Licença CC BY 4.0. Preços em libras esterlinas; base histórica de 2010–2011.
- [scikit-learn: clustering](https://scikit-learn.org/stable/modules/clustering.html).
- [scikit-learn: Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html).
