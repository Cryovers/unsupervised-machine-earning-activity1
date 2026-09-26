"""Segmentação reproduzível da UCI Online Retail. Execute: python atividade1.py."""
from pathlib import Path
from time import perf_counter
from itertools import combinations
import argparse
import hashlib
import json
import platform
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
FEATURES = ['Recency', 'Frequency', 'Monetary']
SEED = 42
URL = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'


def carregar(caminho=None):
    """Baixa somente quando necessário; aceita também CSV/XLSX local."""
    if caminho:
        p = Path(caminho)
    else:
        pasta = ROOT / 'dados'
        pasta.mkdir(exist_ok=True)
        p = pasta / 'Online Retail.xlsx'
        if not p.exists():
            arquivo = pasta / 'online_retail.zip'
            if not arquivo.exists():
                urlretrieve(URL, arquivo)
            with zipfile.ZipFile(arquivo) as z:
                with z.open('Online Retail.xlsx') as src, p.open('wb') as dst:
                    dst.write(src.read())
    df = pd.read_csv(p, dtype={'CustomerID': 'string', 'InvoiceNo': 'string'}) if p.suffix.lower() == '.csv' else pd.read_excel(p)
    return df, hashlib.sha256(p.read_bytes()).hexdigest()


def preparar(df):
    """Uma linha por cliente; exclusões sequenciais para auditoria sem dupla contagem."""
    obrigatorias = {'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'StockCode'}
    if obrigatorias - set(df):
        raise ValueError(f'Colunas ausentes: {sorted(obrigatorias - set(df))}')
    df = df.copy()
    auditoria = {'linhas_originais': len(df)}
    for col in ['Quantity', 'UnitPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['InvoiceDate'] = pd.to_datetime(df.InvoiceDate, errors='coerce')
    df['CustomerID'] = df.CustomerID.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    df['InvoiceNo'] = df.InvoiceNo.astype('string').str.strip()
    referencia = df.InvoiceDate.max().normalize() + pd.Timedelta(days=1)
    if pd.isna(referencia):
        raise ValueError('Nenhuma data válida.')
    def excluir(nome, mascara):
        nonlocal df
        mascara = mascara.fillna(True)
        auditoria[nome] = int(mascara.sum())
        df = df.loc[~mascara].copy()
    excluir('duplicatas', df.duplicated())
    excluir('sem_cliente', df.CustomerID.isna() | df.CustomerID.eq(''))
    excluir('campos_invalidos', df[['InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']].isna().any(axis=1) | df.InvoiceNo.eq('') | ~np.isfinite(df.Quantity) | ~np.isfinite(df.UnitPrice))
    excluir('cancelamentos', df.InvoiceNo.str.upper().str.startswith('C'))
    excluir('quantidade_nao_positiva', df.Quantity.le(0))
    excluir('preco_nao_positivo', df.UnitPrice.le(0))
    df['Total'] = df.Quantity * df.UnitPrice
    if df.empty or not np.isfinite(df.Total).all():
        raise ValueError('Nenhuma venda válida ou valor monetário não finito.')
    rfm = df.groupby('CustomerID').agg(UltimaCompra=('InvoiceDate', 'max'), Frequency=('InvoiceNo', 'nunique'), Monetary=('Total', 'sum'), ProdutosDistintos=('StockCode', 'nunique'), PrimeiraCompra=('InvoiceDate', 'min'))
    rfm['Recency'] = (referencia - rfm.UltimaCompra.dt.normalize()).dt.days
    rfm['TicketMedio'] = rfm.Monetary / rfm.Frequency
    rfm['TempoObservadoDias'] = (referencia - rfm.PrimeiraCompra.dt.normalize()).dt.days
    auditoria.update(linhas_validas=len(df), clientes=len(rfm), referencia=str(referencia.date()), inicio=str(df.InvoiceDate.min()), fim=str(df.InvoiceDate.max()), receita_bruta=float(df.Total.sum()))
    if len(rfm) < 10:
        raise ValueError('São necessários pelo menos 10 clientes válidos para a comparação.')
    X = StandardScaler().fit_transform(np.log1p(rfm[FEATURES]))
    return rfm, X, auditoria


def comparar(X, out):
    registros, modelos, rotulos = [], {}, {}
    def avaliar(nome, modelo, k, familia, metodo=''):
        inicio = perf_counter()
        labels = modelo.fit_predict(X)
        tempo = perf_counter() - inicio
        n = len(np.unique(labels))
        valido = 1 < n < len(X)
        row = dict(modelo=nome, familia=familia, metodo=metodo, k=k, n_clusters=n,
                   silhouette=silhouette_score(X, labels) if valido else np.nan,
                   davies_bouldin=davies_bouldin_score(X, labels) if valido else np.nan,
                   calinski_harabasz=calinski_harabasz_score(X, labels) if valido else np.nan,
                   tempo_segundos=tempo, menor_grupo=float(pd.Series(labels).value_counts(normalize=True).min()),
                   iteracoes=getattr(modelo, 'n_iter_', np.nan), convergiu=getattr(modelo, 'converged_', True))
        if familia == 'KMeans':
            row['inercia'] = modelo.inertia_
            row['convergiu'] = modelo.n_iter_ < modelo.max_iter
        if familia == 'GMM':
            row.update(bic=modelo.bic(X), aic=modelo.aic(X))
        registros.append(row)
        modelos[nome], rotulos[nome] = modelo, labels
    for k in range(2, 9):
        avaliar(f'KMeans_{k}', KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=SEED), k, 'KMeans')
        for method in ['ward', 'complete', 'average']:
            avaliar(f'Hierarquico_{method}_{k}', AgglomerativeClustering(n_clusters=k, linkage=method), k, 'Hierarquico', method)
    for cov in ['full', 'diag']:
        for k in range(1, 9):
            avaliar(f'GMM_{cov}_{k}', GaussianMixture(n_components=k, covariance_type=cov, n_init=5, max_iter=500, reg_covar=1e-5, random_state=SEED), k, 'GMM', cov)
    tabela = pd.DataFrame(registros).set_index('modelo')
    elegiveis = tabela[(tabela.n_clusters > 1) & (tabela.menor_grupo >= .02) & tabela.convergiu]
    if elegiveis.empty:
        raise ValueError('Nenhum modelo atende à regra de pelo menos 2% dos clientes por segmento.')
    # Candidatos de cada abordagem, escolhidos antes da comparação entre famílias.
    candidatos = []
    for familia, metodo in [('KMeans', ''), ('Hierarquico', 'ward'), ('Hierarquico', 'complete'), ('Hierarquico', 'average'), ('GMM', 'full'), ('GMM', 'diag')]:
        sub = elegiveis[(elegiveis.familia == familia) & (elegiveis.metodo == metodo)]
        if len(sub):
            candidatos.append(sub.bic.idxmin() if familia == 'GMM' else sub.silhouette.idxmax())
    selecao = tabela.loc[candidatos].copy()
    selecao['ranking_medio'] = (selecao.silhouette.rank(ascending=False) + selecao.davies_bouldin.rank() + selecao.calinski_harabasz.rank(ascending=False)) / 3
    selecao = selecao.sort_values(['ranking_medio', 'silhouette'], ascending=[True, False])
    escolhido = selecao.index[0]
    km = elegiveis[elegiveis.familia == 'KMeans'].silhouette.idxmax()
    k = int(tabela.loc[km, 'k'])
    runs, stability = [], []
    for seed in range(10):
        m = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=seed).fit(X)
        runs.append(m.labels_)
        stability.append(dict(semente=seed, inercia=m.inertia_, iteracoes=m.n_iter_, convergiu=m.n_iter_ < m.max_iter, ari_referencia=adjusted_rand_score(rotulos[km], m.labels_)))
    pd.DataFrame(stability).to_csv(out / 'estabilidade_kmeans.csv', index=False)
    tabela.to_csv(out / 'comparacao_modelos.csv')
    selecao.to_csv(out / 'selecao_modelos.csv')
    gmm = tabela[(tabela.familia == 'GMM') & tabela.convergiu].bic.idxmin()
    probs = modelos[gmm].predict_proba(X)
    pd.DataFrame(probs, columns=[f'componente_{i}' for i in range(probs.shape[1])]).to_csv(out / 'probabilidades_gmm.csv', index=False)
    info = dict(escolhido=escolhido, kmeans=km, gmm_bic_global=gmm, gmm_aic_global=tabela[tabela.familia.eq('GMM') & tabela.convergiu].aic.idxmin(), gmm_incertos=int((probs.max(axis=1) < .7).sum()), ari_medio=float(np.mean([adjusted_rand_score(a,b) for a,b in combinations(runs,2)])))
    return tabela, rotulos[escolhido], info


def perfilar(rfm, labels):
    rfm = rfm.copy()
    rfm['Cluster'] = labels
    perfil = rfm.groupby('Cluster').agg(Clientes=('Monetary', 'size'), Recency=('Recency', 'mean'), Frequency=('Frequency', 'mean'), Monetary=('Monetary', 'mean'), Receita=('Monetary', 'sum'), TicketMedio=('TicketMedio', 'mean'), ProdutosDistintos=('ProdutosDistintos', 'mean'))
    perfil['PercentualClientes'] = 100 * perfil.Clientes / len(rfm)
    perfil['PercentualReceita'] = 100 * perfil.Receita / perfil.Receita.sum()
    rlim, flim, mlim = rfm[FEATURES].median()
    nomes, acoes = {}, {}
    for c, p in perfil.iterrows():
        if p.Recency > rlim and p.Monetary >= mlim:
            nome, acao = 'Valiosos em risco', 'Contato personalizado de reativação; medir recompra incremental em 60 dias.'
        elif p.Recency > rlim:
            nome, acao = 'Inativos de menor valor', 'Reativação de baixo custo; medir margem incremental e descadastros.'
        elif p.Frequency >= flim and p.Monetary >= mlim:
            nome, acao = 'Campeões e fiéis', 'Benefícios de fidelidade e venda complementar; medir retenção e margem em 90 dias.'
        else:
            nome, acao = 'Recentes em desenvolvimento', 'Jornada de segunda compra; medir conversão em 30 dias.'
        nomes[c], acoes[c] = f'{nome} (C{c})', acao
    perfil['Segmento'] = pd.Series(nomes)
    perfil['Acao'] = pd.Series(acoes)
    rfm['Segmento'] = rfm.Cluster.map(nomes)
    return rfm, perfil


def visualizar(rfm, X, tabela, perfil, info, out):
    figs = {}
    km = tabela[tabela.familia.eq('KMeans')].sort_values('k')
    figs['cotovelo'] = px.line(km, x='k', y='inercia', markers=True, title='K-Means: método do cotovelo')
    # Distância à reta entre extremos da curva normalizada: heurística, não prova de ótimo.
    xx = (km.k.to_numpy() - km.k.min()) / (km.k.max() - km.k.min())
    yy = (km.inercia.to_numpy() - km.inercia.min()) / (km.inercia.max() - km.inercia.min())
    info['k_cotovelo'] = int(km.k.iloc[np.argmax(np.abs(xx + yy - 1))])
    figs['silhouette'] = px.line(tabela[tabela.familia.ne('GMM')].reset_index().assign(serie=lambda t:t.familia + ' ' + t.metodo), x='k', y='silhouette', color='serie', markers=True, title='Silhouette: maior é melhor')
    gm = tabela[tabela.familia.eq('GMM')].reset_index().melt(id_vars=['k','metodo'], value_vars=['bic','aic'], var_name='Criterio', value_name='Valor')
    figs['gmm'] = px.line(gm, x='k', y='Valor', color='metodo', line_dash='Criterio', markers=True, title='GMM: BIC e AIC (menor é melhor)')
    pca = PCA(n_components=3).fit(X)
    pontos = pd.DataFrame(pca.transform(X), columns=['PC1','PC2','PC3'], index=rfm.index).join(rfm[['Segmento'] + FEATURES])
    variance = pca.explained_variance_ratio_
    info['pca_variancia'] = variance.tolist()
    info['pca_componentes_80'] = int(np.searchsorted(np.cumsum(variance), .8) + 1)
    cargas = pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1','PC2','PC3'])
    cargas.to_csv(out / 'pca_cargas.csv')
    centros = pontos.groupby('Segmento')[['PC1','PC2','PC3']].mean()
    figs['pca2d'] = px.scatter(pontos, x='PC1', y='PC2', color='Segmento', hover_data=FEATURES, opacity=.6, title=f'PCA 2D — {sum(variance[:2]):.1%} da variância')
    figs['pca2d'].add_trace(go.Scatter(x=centros.PC1, y=centros.PC2, mode='markers', marker=dict(symbol='x', size=15, color='black'), text=centros.index, name='Centroides projetados'))
    figs['pca3d'] = px.scatter_3d(pontos, x='PC1', y='PC2', z='PC3', color='Segmento', hover_data=FEATURES, opacity=.6, title='PCA 3D — 100% da variância RFM')
    figs['pca3d'].add_trace(go.Scatter3d(x=centros.PC1, y=centros.PC2, z=centros.PC3, mode='markers', marker=dict(symbol='x', size=9, color='black'), text=centros.index, name='Centroides projetados'))
    medias = pd.DataFrame(X, index=rfm.index, columns=FEATURES).groupby(rfm.Cluster).mean()
    normalized = pd.DataFrame(MinMaxScaler().fit_transform(medias), index=medias.index, columns=FEATURES)
    radar = go.Figure()
    for c, row in normalized.iterrows():
        radar.add_trace(go.Scatterpolar(r=row.tolist()+[row.iloc[0]], theta=FEATURES+[FEATURES[0]], fill='toself', name=perfil.loc[c,'Segmento']))
    radar.update_layout(title='Perfis RFM: médias log-padronizadas, reescaladas entre segmentos', polar=dict(radialaxis=dict(range=[0,1])))
    figs['radar'] = radar
    figs['heatmap'] = go.Figure(go.Heatmap(z=medias.values, x=FEATURES, y=perfil.Segmento, text=perfil[FEATURES].round(1).astype(str).values, texttemplate='%{text}', colorscale='RdBu_r', zmid=0, colorbar_title='Média z(log1p)'))
    figs['heatmap'].update_layout(title='Cor: média padronizada; anotação: média nas unidades originais')
    figs['clientes'] = px.bar(perfil, x='Segmento', y='Clientes', color='Segmento', title='Clientes por segmento')
    figs['valor'] = px.box(rfm, x='Segmento', y='Monetary', color='Segmento', log_y=True, title='Valor monetário por cliente — GBP, escala logarítmica')
    figs['receita'] = px.pie(perfil, names='Segmento', values='Receita', title='Participação na receita bruta observada (GBP)')
    # Árvore de todos os clientes; truncamento é apenas visual.
    hier = tabela[tabela.familia.eq('Hierarquico') & tabela.menor_grupo.ge(.02)]
    best = hier.loc[hier.silhouette.idxmax()]
    Z = linkage(X, method=best.metodo)
    k = int(best.k)
    limiar = float((Z[-k,2] + Z[-k+1,2]) / 2)
    info['hierarquia'] = dict(metodo=best.metodo, k=k, corte=limiar, salto=float(Z[-k+1,2]-Z[-k,2]))
    fig, ax = plt.subplots(figsize=(13,5))
    dendrogram(Z, truncate_mode='lastp', p=30, color_threshold=limiar, ax=ax)
    ax.axhline(limiar, color='red', linestyle='--', label=f'Corte k={k}, altura={limiar:.2f}')
    ax.set(title=f'Dendrograma {best.metodo} — árvore completa, últimos 30 ramos', xlabel='Ramo / quantidade de clientes entre parênteses', ylabel='Distância de ligação')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'dendrograma.png', dpi=160)
    plt.close(fig)
    for nome, f in figs.items():
        f.update_layout(template='plotly_white', margin=dict(l=55,r=35,t=85,b=80))
        f.write_json(out / f'{nome}.json')
    import base64
    png = base64.b64encode((out / 'dendrograma.png').read_bytes()).decode()
    cards = ''.join(f'<section>{f.to_html(full_html=False, include_plotlyjs=True if i==0 else False)}</section>' for i,f in enumerate(figs.values()))
    html = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Atividade 1 | Segmentação de clientes</title><style>body{{font-family:system-ui;background:#f1f5f9;color:#132238;margin:0}}main{{max-width:1250px;margin:auto;padding:24px}}section{{background:white;border-radius:12px;padding:14px;margin:24px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #ddd}}img{{max-width:100%}}</style><main><h1>Segmentação de clientes</h1><p>UCI Online Retail · {len(rfm):,} clientes · Modelo: {info['escolhido']} · Receita bruta: £{rfm.Monetary.sum():,.2f}</p><p>Explore os pontos, gire o gráfico 3D e clique nas legendas para mostrar ou ocultar segmentos. Recency menor indica compra mais recente. Receita não equivale a lucro.</p><section><h2>Segmentos e ações</h2>{perfil.round(2).to_html(index=False)}</section>{cards}<section><h2>Estrutura hierárquica</h2><img src="data:image/png;base64,{png}" alt="Dendrograma dos clientes"><p>Corte escolhido por silhouette entre k=2 e 8 e grupos com pelo menos 2% dos clientes. Ramos abaixo do corte são mais semelhantes.</p></section><section><h2>Comparação de modelos</h2>{tabela.round(4).to_html()}</section><p>Fonte: Chen (2015), <a href="https://doi.org/10.24432/C5BW33">UCI Online Retail</a>, CC BY 4.0. Dados históricos de 2010–2011.</p></main></html>'''
    (out / 'dashboard.html').write_text(html, encoding='utf-8')
    return cargas


def relatorio(rfm, perfil, tabela, info, auditoria, cargas, out):
    win = tabela.loc[info['escolhido']]
    maior = perfil.Receita.idxmax()
    risco = perfil.Recency.idxmax()
    segmentos = '\n'.join(f"- **{p.Segmento}**: {int(p.Clientes)} clientes ({p.PercentualClientes:.1f}%), {p.PercentualReceita:.1f}% da receita; recência média {p.Recency:.0f} dias, {p.Frequency:.1f} pedidos e £{p.Monetary:,.0f} por cliente. {p.Acao}" for _,p in perfil.iterrows())
    texto = f'''# Relatório executivo — Segmentação de clientes

## 1. Sumário executivo

- A base contém {auditoria['linhas_originais']:,} linhas. Após a limpeza, {auditoria['linhas_validas']:,} linhas de compras válidas representam {len(rfm):,} clientes e £{rfm.Monetary.sum():,.2f} de receita bruta observada.
- O modelo selecionado foi **{info['escolhido']}**, com {int(win.n_clusters)} segmentos. A maior concentração de receita está em **{perfil.loc[maior,'Segmento']}**, responsável por {perfil.loc[maior,'PercentualReceita']:.1f}% do total.
- O segmento **{perfil.loc[risco,'Segmento']}** apresenta a maior recência média: {perfil.loc[risco,'Recency']:.0f} dias sem compra. Isso sinaliza inatividade relativa, sem comprovar abandono definitivo.

Três prioridades: (1) proteger clientes de maior valor com fidelização e atendimento direcionado; (2) testar reativação dos segmentos inativos; (3) estimular uma nova compra nos segmentos de menor frequência. Validar todas as intervenções com grupos de controle antes de ampliar investimentos.

## 2. Metodologia e validação

Foi utilizada a base UCI Online Retail, com vendas de dezembro de 2010 a dezembro de 2011, em libras esterlinas. Foram removidas duplicatas exatas, linhas sem cliente, campos essenciais inválidos, cancelamentos e quantidades/preços não positivos, nessa ordem. A auditoria registra as exclusões sem dupla contagem. A receita mede vendas positivas identificadas, não receita líquida de devoluções, margem ou lucro.

Cada cliente é representado por recência em dias até {auditoria['referencia']}, frequência de notas fiscais distintas e soma monetária. Foram aplicados log1p e StandardScaler para reduzir assimetria e equilibrar escalas, preservando grandes compradores. Ticket médio, diversidade de produtos e tempo observado complementam a interpretação, mas não entram no agrupamento.

A busca compara K-Means e hierárquico ward/complete/average com k entre 2 e 8; GMM full/diag com 1 a 8 componentes, cinco inicializações e regularização. Candidatos operacionais exigem convergência e no mínimo 2% dos clientes em cada grupo. Dentro de cada abordagem, foi utilizado o maior silhouette, ou o menor BIC para GMM. Entre candidatos, vence a menor média dos rankings de silhouette, Davies–Bouldin e Calinski–Harabasz, com desempate por silhouette. É uma regra explícita de compromisso, não um ótimo universal.

O vencedor obteve silhouette **{win.silhouette:.3f}**, Davies–Bouldin **{win.davies_bouldin:.3f}** e Calinski–Harabasz **{win.calinski_harabasz:.1f}**. Métricas usam todos os clientes no mesmo espaço RFM transformado. Tempos medem ajuste e atribuição, excluindo cálculo de métricas. Não há rótulos verdadeiros; validação interna não demonstra retorno financeiro.

## 3. Resultados e insights

{segmentos}

O cotovelo geométrico do K-Means sugere k={info['k_cotovelo']}; a escolha por silhouette e tamanho mínimo foi {info['kmeans']}. Discordância evidencia que compactação e separação são critérios diferentes. Dez sementes com 20 inicializações produziram ARI médio entre pares de {info['ari_medio']:.3f}; as iterações e inércias estão no arquivo de estabilidade. Estabilidade entre sementes não garante estabilidade ao longo do tempo.

No hierárquico, a configuração operacional com melhor silhouette usa {info['hierarquia']['metodo']}, k={info['hierarquia']['k']}, com corte a {info['hierarquia']['corte']:.2f} e salto de ligação de {info['hierarquia']['salto']:.2f}. A linha vermelha separa os ramos; o desenho exibe os últimos 30 ramos de uma árvore calculada com todos os clientes. Complete/average podem isolar grandes compradores, razão para explicitar a restrição de tamanho.

O maior segmento contém {perfil.PercentualClientes.max():.1f}% dos clientes: essa concentração reduz a granularidade comercial e pode ocultar subgrupos. Avaliar k=3 a 8 em testes de campanhas é um próximo passo, mesmo que as métricas internas sejam menores.

O menor BIC global foi de {info['gmm_bic_global']}; o menor AIC foi de {info['gmm_aic_global']}. Esses mínimos são reportados mesmo quando não atendem à regra operacional. No GMM de menor BIC, {info['gmm_incertos']} clientes têm probabilidade máxima inferior a 70%: casos ambíguos que merecem campanhas menos específicas. Probabilidades são pertinências do modelo, não probabilidades de compra. Um mínimo em oito componentes está na fronteira da busca e não comprova ótimo global.

PCA em duas dimensões retém {sum(info['pca_variancia'][:2]):.1%} da variância; {info['pca_componentes_80']} componentes atingem 80%, e a visualização 3D preserva 100%. PCA é usado apenas para visualizar. As cargas assinadas estão no notebook: sua magnitude mostra quais variáveis influenciam cada eixo; o sinal dos eixos é arbitrário. Centroides são médias dos grupos projetadas, inclusive quando o algoritmo não possui centroides próprios.

## 4. Recomendações, limitações e próximos passos

Marketing/CRM deve conduzir testes aleatorizados por segmento, com controle sem campanha, comparando recompra, receita por cliente e margem incremental em 30, 60 e 90 dias. Finanças deve definir um teto de custo por cliente a partir da margem disponível. Não é possível estimar ROI sem custos, descontos e efeito causal. Priorizar segmentos combinando tamanho, valor e recência; evitar oferecer descontos generalizados aos compradores já fiéis.

Clientes recentes não são necessariamente novos: a janela limita a observação. Os nomes são regras descritivas baseadas nas médias dos segmentos e medianas globais, não classes RFM 555 nem diagnósticos individuais. A empresa inclui atacadistas; valores extremos podem ser legítimos. Cancelamentos excluídos, compradores anônimos, sazonalidade e dados históricos limitam a generalização. Duplicatas exatas podem representar compras legítimas; a decisão deve ser auditada com a origem. Como evolução, comparar janelas temporais, incluir devoluções líquidas, avaliar estabilidade por reamostragem e revisar segmentos mensalmente.

Fonte: Chen, D. (2015). [Online Retail, UCI](https://doi.org/10.24432/C5BW33), CC BY 4.0.
'''
    assert len(texto.split()) <= 1500, 'Relatório ultrapassou 1500 palavras.'
    (out / 'relatorio_executivo.md').write_text(texto, encoding='utf-8')


def executar(caminho=None, destino=None):
    out = Path(destino) if destino else ROOT / 'resultados'
    out.mkdir(parents=True, exist_ok=True)
    print('Carregando e preparando dados...', flush=True)
    bruto, sha = carregar(caminho)
    rfm, X, auditoria = preparar(bruto)
    print(f'Comparando modelos para {len(rfm)} clientes...', flush=True)
    with threadpool_limits(limits=1):
        tabela, labels, info = comparar(X, out)
        rfm, perfil = perfilar(rfm, labels)
        cargas = visualizar(rfm, X, tabela, perfil, info, out)
    probs = pd.read_csv(out / 'probabilidades_gmm.csv')
    probs.insert(0, 'CustomerID', rfm.index.to_numpy())
    probs['ProbabilidadeMaxima'] = probs.drop(columns='CustomerID').max(axis=1)
    probs['AtribuicaoIncerta'] = probs.ProbabilidadeMaxima.lt(.7)
    probs.to_csv(out / 'probabilidades_gmm.csv', index=False)
    rfm.to_csv(out / 'clientes_segmentados.csv')
    perfil.to_csv(out / 'perfis_segmentos.csv')
    (out / 'auditoria.json').write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding='utf-8')
    info.update(sha256_dados=sha, semente=SEED, python=platform.python_version())
    (out / 'resumo.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    relatorio(rfm, perfil, tabela, info, auditoria, cargas, out)
    print(f"Concluído: {info['escolhido']}. Arquivos em {out}", flush=True)
    return rfm, perfil, tabela, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dados', type=Path, help='CSV ou XLSX no esquema Online Retail; por padrão baixa a base UCI.')
    parser.add_argument('--saida', type=Path, default=ROOT / 'resultados')
    args = parser.parse_args()
    executar(args.dados, args.saida)
