import io
import html
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)

st.set_page_config(
    page_title="Odin | Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>

        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            color: #667085;
            font-size: 1rem;
            margin-bottom: 20px;
        }

        /* Card adaptável a tema claro/escuro */
        div[data-testid="stMetric"] {
            background-color: rgba(127, 127, 127, 0.08);
            border: 1px solid rgba(127, 127, 127, 0.25);
            padding: 14px;
            border-radius: 12px;
        }

        .alert-box {
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .footer {
            color: #667085;
            font-size: 0.82rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">📊 Odin — Plataforma Analítico-Preditiva</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Monitoramento de incidentes, previsão de volume, tendências, '
    'risco de OLA e geração automática de relatórios.'
    '</div>',
    unsafe_allow_html=True,
)

REQUIRED_COLUMNS = [
    "data",
    "produto",
    "prioridade",
    "quantidade",
    "equipe",
]

PRIORITY_WEIGHT = {
    "P1": 4,
    "P2": 3,
    "P3": 2,
    "P4": 1,
}

def normalizar_dados(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # Normalização dos nomes das colunas
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Verificação
    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing)
        )

    # Data
    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    )

    # Textos
    df["produto"] = (
        df["produto"]
        .astype(str)
        .str.strip()
    )

    df["prioridade"] = (
        df["prioridade"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["equipe"] = (
        df["equipe"]
        .astype(str)
        .str.strip()
    )

    # Quantidade
    df["quantidade"] = pd.to_numeric(
        df["quantidade"],
        errors="coerce",
    )

    df["quantidade"] = (
        df["quantidade"]
        .fillna(0)
        .clip(lower=0)
    )

    # Remover registros inválidos
    df = df.dropna(
        subset=[
            "data",
            "produto",
            "prioridade",
            "equipe",
        ]
    )

    # Consolidar registros duplicados
    df = (
        df.groupby(
            [
                "data",
                "produto",
                "prioridade",
                "equipe",
            ],
            as_index=False,
        )["quantidade"]
        .sum()
    )

    return df

@st.cache_data
def gerar_dados_simulados(
    dias: int = 120,
) -> pd.DataFrame:

    rng = np.random.default_rng(42)

    datas = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=dias,
        freq="D",
    )

    produtos = [
        "E-mail Corporativo",
        "Hospedagem WordPress",
        "Cloud VPS",
        "Registro de Domínio",
        "SSL",
    ]

    equipes = {
        "E-mail Corporativo": "E-mail",
        "Hospedagem WordPress": "Hospedagem",
        "Cloud VPS": "Infra Cloud",
        "Registro de Domínio": "Domínios",
        "SSL": "Segurança",
    }

    prioridades = [
        "P2",
        "P3",
    ]

    base = {
        ("E-mail Corporativo", "P2"): 12,
        ("E-mail Corporativo", "P3"): 8,

        ("Hospedagem WordPress", "P2"): 8,
        ("Hospedagem WordPress", "P3"): 10,

        ("Cloud VPS", "P2"): 6,
        ("Cloud VPS", "P3"): 5,

        ("Registro de Domínio", "P2"): 3,
        ("Registro de Domínio", "P3"): 2,

        ("SSL", "P2"): 2,
        ("SSL", "P3"): 1,
    }

    registros = []

    for data in datas:

        dia = (data - datas[0]).days

        # Sazonalidade semanal
        sazonalidade = (
            1
            + 0.10
            * np.sin(dia / 7)
        )

        for produto in produtos:

            for prioridade in prioridades:

                media = base[
                    (produto, prioridade)
                ]

                # Tendência artificial no E-mail/P2
                if (
                    produto == "E-mail Corporativo"
                    and prioridade == "P2"
                    and data >= datas[-30]
                ):
                    dias_tendencia = (
                        data - datas[-30]
                    ).days

                    media += (
                        dias_tendencia * 0.16
                    )

                media *= sazonalidade

                quantidade = rng.poisson(
                    max(media, 0.5)
                )

                registros.append(
                    [
                        data,
                        produto,
                        prioridade,
                        int(quantidade),
                        equipes[produto],
                    ]
                )

    df = pd.DataFrame(
        registros,
        columns=REQUIRED_COLUMNS,
    )

    return normalizar_dados(df)


try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    RandomForestRegressor = None


def _criar_serie_diaria(df, produto=None):
    dados = df.copy()
    if produto and produto != "Todos":
        dados = dados[dados["produto"] == produto]
    return (dados.groupby("data")["quantidade"].sum()
            .asfreq("D", fill_value=0).astype(float))


def _criar_atributos_temporais(serie):
    base = pd.DataFrame({"y": serie.astype(float)})
    for lag in [1, 2, 3, 7, 14]:
        base[f"lag_{lag}"] = base["y"].shift(lag)
    for janela in [3, 7, 14]:
        base[f"media_{janela}"] = base["y"].shift(1).rolling(janela).mean()
        base[f"std_{janela}"] = base["y"].shift(1).rolling(janela).std()
    base["tendencia_7"] = base["media_3"] - base["media_7"]
    idx = base.index
    base["dia_semana"] = idx.dayofweek
    base["fim_de_semana"] = (idx.dayofweek >= 5).astype(int)
    base["dia_mes"] = idx.day
    base["mes"] = idx.month
    base["semana_ano"] = idx.isocalendar().week.astype(int)
    base["sin_semana"] = np.sin(2 * np.pi * base["dia_semana"] / 7)
    base["cos_semana"] = np.cos(2 * np.pi * base["dia_semana"] / 7)
    base["sin_mes"] = np.sin(2 * np.pi * base["mes"] / 12)
    base["cos_mes"] = np.cos(2 * np.pi * base["mes"] / 12)
    return base


def _colunas_modelo():
    return ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
            "media_3", "media_7", "media_14", "std_3", "std_7", "std_14",
            "tendencia_7", "dia_semana", "fim_de_semana", "dia_mes", "mes",
            "semana_ano", "sin_semana", "cos_semana", "sin_mes", "cos_mes"]


def _prever_random_forest(serie, dias=7):
    if RandomForestRegressor is None:
        raise ImportError("scikit-learn não está instalado. Execute: pip install scikit-learn")
    if len(serie) < 30:
        raise ValueError("São necessários pelo menos 30 dias de histórico para o modelo de Machine Learning.")

    atributos = _criar_atributos_temporais(serie)
    features = _colunas_modelo()
    base = atributos.dropna().copy()
    if len(base) < 20:
        raise ValueError("Histórico insuficiente após a criação das variáveis preditivas.")

    teste_n = max(7, min(14, int(len(base) * 0.20)))
    treino, teste = base.iloc[:-teste_n], base.iloc[-teste_n:]

    modelo = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2,
                                   max_features="sqrt", random_state=42, n_jobs=-1)
    modelo.fit(treino[features], treino["y"])
    pred_teste = np.maximum(0, modelo.predict(teste[features]))

    mae = float(mean_absolute_error(teste["y"], pred_teste))
    rmse = float(np.sqrt(mean_squared_error(teste["y"], pred_teste)))
    r2 = float(r2_score(teste["y"], pred_teste)) if len(teste) > 1 else float("nan")

    # Modelo final usa todo o histórico disponível.
    modelo_final = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2,
                                         max_features="sqrt", random_state=42, n_jobs=-1)
    modelo_final.fit(base[features], base["y"])

    serie_ext = serie.copy()
    previsoes = []
    erro = max(rmse, float(serie.diff().dropna().std() or 1.0))

    for _ in range(dias):
        proxima = pd.Timestamp(serie_ext.index.max()) + pd.Timedelta(days=1)


        serie_ext.loc[proxima] = np.nan
        serie_ext = serie_ext.sort_index()

        futuro_atributos = _criar_atributos_temporais(serie_ext)
        linha = futuro_atributos.loc[[proxima], features]

        # Segurança extra: se alguma feature não puder ser calculada
        # por falta de histórico, interrompemos com uma mensagem clara.
        if linha[features].isna().any().any():
            raise ValueError(
                "Não foi possível calcular todas as variáveis da previsão. "
                "Utilize pelo menos 30 dias de histórico contínuo."
            )

        pred = max(0.0, float(modelo_final.predict(linha)[0]))
        previsoes.append(pred)


        serie_ext.loc[proxima] = pred

    futuro = pd.date_range(serie.index.max() + timedelta(days=1), periods=dias, freq="D")
    previsoes_np = np.array(previsoes)
    margem = np.full(dias, 1.96 * erro)

    previsao_df = pd.DataFrame({
        "data": futuro,
        "previsao": previsoes_np,
        "limite_inferior": np.maximum(0, previsoes_np - margem),
        "limite_superior": previsoes_np + margem,
    })

    media_7 = float(serie.tail(7).mean())
    desvio_7 = float(serie.tail(7).std() or 1.0)
    limiar_pico = max(media_7 * 1.20, media_7 + desvio_7)
    previsao_df["risco_pico"] = ((previsao_df["previsao"] - limiar_pico) /
                                  max(desvio_7, 1.0) * 50 + 50).clip(0, 99).round(1)
    previsao_df["classificacao_risco"] = pd.cut(
        previsao_df["risco_pico"], bins=[-1, 30, 60, 80, 100],
        labels=["Baixo", "Moderado", "Alto", "Crítico"])

    importancia = pd.DataFrame({"variavel": features,
                                "importancia": modelo_final.feature_importances_}).sort_values(
                                    "importancia", ascending=False)
    metricas = {"MAE": mae, "RMSE": rmse, "R2": r2,
                "amostras_treino": len(treino), "amostras_teste": len(teste),
                "limiar_pico": limiar_pico}
    return previsao_df, metricas, importancia


def prever_volume(df, produto=None, dias=7):
    serie = _criar_serie_diaria(df, produto)
    previsao_df, metricas, importancia = _prever_random_forest(serie, dias)
    return serie, previsao_df, metricas, importancia


def calcular_risco_ola(
    df: pd.DataFrame,
) -> pd.DataFrame:

    dados = df.copy()

    # Peso da prioridade
    dados["peso_prioridade"] = (
        dados["prioridade"]
        .map(PRIORITY_WEIGHT)
        .fillna(1)
    )

    # Volume ponderado
    dados["volume_ponderado"] = (
        dados["quantidade"]
        * dados["peso_prioridade"]
    )

    data_max = dados["data"].max()

    inicio_recente = (
        data_max
        - timedelta(days=6)
    )

    inicio_anterior = (
        data_max
        - timedelta(days=13)
    )

    # Últimos 7 dias
    recente = dados[
        dados["data"]
        >= inicio_recente
    ]

    # 7 dias anteriores
    anterior = dados[
        (dados["data"]
         >= inicio_anterior)
        &
        (dados["data"]
         < inicio_recente)
    ]

    rec = (
        recente
        .groupby("equipe")[
            "volume_ponderado"
        ]
        .sum()
        .rename("recente")
    )

    ant = (
        anterior
        .groupby("equipe")[
            "volume_ponderado"
        ]
        .sum()
        .rename("anterior")
    )

    result = pd.concat(
        [
            rec,
            ant,
        ],
        axis=1,
    ).fillna(0)

    # Variação
    result["variacao"] = (
        (
            result["recente"]
            - result["anterior"]
        )
        /
        result["anterior"].replace(
            0,
            np.nan,
        )
    ) * 100

    result["variacao"] = (
        result["variacao"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0)
    )

    # Média diária
    result["media_diaria"] = (
        result["recente"] / 7
    )


    if len(result) > 1:

        componente_volume = (
            result["media_diaria"]
            .rank(pct=True)
            * 55
        )

    else:

        componente_volume = pd.Series(
            55,
            index=result.index,
        )

 
    componente_tendencia = (
        (
            result["variacao"]
            .clip(-50, 100)
            + 50
        )
        / 150
        * 45
    )


    result["score"] = (
        componente_volume
        + componente_tendencia
    )

    result["risco_ola"] = (
        result["score"]
        .clip(
            5,
            95,
        )
        .round(1)
    )



    result["classificacao"] = pd.cut(
        result["risco_ola"],
        bins=[
            0,
            30,
            60,
            100,
        ],
        labels=[
            "Baixo",
            "Moderado",
            "Alto",
        ],
        include_lowest=True,
    )

  
    result = result.reset_index()

    return result.sort_values(
        "risco_ola",
        ascending=False,
    )



def gerar_alertas(
    df: pd.DataFrame,
    previsao_df: pd.DataFrame,
    risco_df: pd.DataFrame,
    produto_foco: str,
):

    alertas = []

    data_max = df["data"].max()



    ultimos_7 = df[
        df["data"]
        > data_max
        - timedelta(days=7)
    ]

    anteriores_7 = df[
        (
            df["data"]
            <= data_max
            - timedelta(days=7)
        )
        &
        (
            df["data"]
            > data_max
            - timedelta(days=14)
        )
    ]

    if produto_foco != "Todos":

        volume_recente = (
            ultimos_7[
                ultimos_7["produto"]
                == produto_foco
            ]["quantidade"]
            .sum()
        )

        volume_anterior = (
            anteriores_7[
                anteriores_7["produto"]
                == produto_foco
            ]["quantidade"]
            .sum()
        )

    else:

        volume_recente = (
            ultimos_7["quantidade"]
            .sum()
        )

        volume_anterior = (
            anteriores_7["quantidade"]
            .sum()
        )

    if volume_anterior > 0:

        variacao = (
            (
                volume_recente
                - volume_anterior
            )
            / volume_anterior
            * 100
        )

    else:

        variacao = 0

    if variacao >= 20:

        alertas.append(
            {
                "nivel": "Crítico",
                "titulo": "Tendência de alta",
                "mensagem": (
                    f"{produto_foco} apresentou "
                    f"aumento de {variacao:.1f}% "
                    f"nos últimos 7 dias."
                ),
            }
        )

    elif variacao >= 10:

        alertas.append(
            {
                "nivel": "Atenção",
                "titulo": "Tendência de crescimento",
                "mensagem": (
                    f"{produto_foco} apresenta "
                    f"crescimento de {variacao:.1f}% "
                    f"em relação ao período anterior."
                ),
            }
        )


    if len(previsao_df) > 0:

        media_recente = (
            ultimos_7["quantidade"]
            .mean()
        )

        previsao_d1 = (
            previsao_df
            .iloc[0]["previsao"]
        )

        if (
            media_recente > 0
            and previsao_d1
            > media_recente * 1.20
        ):

            alertas.append(
                {
                    "nivel": "Atenção",
                    "titulo": "Pico previsto",
                    "mensagem": (
                        f"A previsão para D+1 é de "
                        f"{previsao_d1:.0f} incidentes, "
                        f"acima da média recente de "
                        f"{media_recente:.1f}."
                    ),
                }
            )


    for _, row in risco_df.head(3).iterrows():

        if row["risco_ola"] >= 70:

            alertas.append(
                {
                    "nivel": "Crítico",
                    "titulo": "Risco elevado de OLA",
                    "mensagem": (
                        f"A equipe {row['equipe']} "
                        f"apresenta risco estimado "
                        f"de {row['risco_ola']:.1f}%."
                    ),
                }
            )



    if not alertas:

        alertas.append(
            {
                "nivel": "Normal",
                "titulo": "Cenário estável",
                "mensagem": (
                    "Os indicadores analisados não "
                    "apresentaram anomalias relevantes."
                ),
            }
        )

    return alertas


def grafico_historico_previsao(
    df,
    previsao_df,
    produto_foco,
):

    if produto_foco != "Todos":

        dados = df[
            df["produto"] == produto_foco
        ]

    else:

        dados = df

    serie = (
        dados
        .groupby("data")["quantidade"]
        .sum()
        .reset_index()
    )

    fig = go.Figure()

    # Histórico
    fig.add_trace(
        go.Scatter(
            x=serie["data"],
            y=serie["quantidade"],
            name="Histórico",
            mode="lines+markers",
            line=dict(
                width=2,
            ),
        )
    )

    # Previsão
    fig.add_trace(
        go.Scatter(
            x=previsao_df["data"],
            y=previsao_df["previsao"],
            name="Previsão",
            mode="lines+markers",
            line=dict(
                width=3,
                dash="dash",
            ),
        )
    )

    # Intervalo
    fig.add_trace(
        go.Scatter(
            x=(
                list(previsao_df["data"])
                + list(
                    previsao_df["data"]
                    [::-1]
                )
            ),
            y=(
                list(
                    previsao_df[
                        "limite_superior"
                    ]
                )
                +
                list(
                    previsao_df[
                        "limite_inferior"
                    ][::-1]
                )
            ),
            fill="toself",
            fillcolor=(
                "rgba(99,110,250,0.15)"
            ),
            line=dict(
                color="rgba(255,255,255,0)"
            ),
            name="Intervalo 95%",
        )
    )

    fig.update_layout(
        title=(
            "Histórico + previsão — "
            f"{produto_foco}"
        ),
        xaxis_title="Data",
        yaxis_title="Incidentes",
        template="plotly_white",
        hovermode="x unified",
        height=450,
        legend=dict(
            orientation="h",
            y=1.1,
        ),
    )

    return fig


def grafico_pareto(df):

    dados = (
        df.groupby("produto")[
            "quantidade"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .reset_index()
    )

    total = dados["quantidade"].sum()

    if total > 0:

        dados["acumulado"] = (
            dados["quantidade"]
            .cumsum()
            / total
            * 100
        )

    else:

        dados["acumulado"] = 0

    fig = go.Figure()

    fig.add_bar(
        x=dados["produto"],
        y=dados["quantidade"],
        name="Incidentes",
    )

    fig.add_scatter(
        x=dados["produto"],
        y=dados["acumulado"],
        name="% acumulado",
        mode="lines+markers",
        yaxis="y2",
    )

    fig.update_layout(
        title="Pareto de incidentes por produto",
        yaxis=dict(
            title="Incidentes"
        ),
        yaxis2=dict(
            title="% acumulado",
            overlaying="y",
            side="right",
            range=[
                0,
                110,
            ],
        ),
        template="plotly_white",
        height=420,
    )

    return fig

def grafico_prioridades(df):

    dados = (
        df.groupby(
            [
                "produto",
                "prioridade",
            ]
        )["quantidade"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        dados,
        x="produto",
        y="quantidade",
        color="prioridade",
        barmode="group",
        title=(
            "Incidentes por produto "
            "e prioridade"
        ),
        template="plotly_white",
    )

    fig.update_layout(
        height=420
    )

    return fig



def grafico_risco_ola(
    risco_df,
):

    # IMPORTANTE:
    # A coluna real é "equipe", não "Equipe".
    dados = (
        risco_df
        .sort_values(
            "risco_ola",
            ascending=True,
        )
        .copy()
    )

    fig = px.bar(
        dados,
        x="risco_ola",
        y="equipe",
        orientation="h",
        color="risco_ola",
        text="risco_ola",
        color_continuous_scale="RdYlGn_r",
        title=(
            "Índice estimado "
            "de risco de OLA"
        ),
        template="plotly_white",
        labels={
            "risco_ola": "Risco OLA (%)",
            "equipe": "Equipe",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    max_value = max(
        100,
        float(
            dados["risco_ola"].max()
        )
        + 10,
    )

    fig.update_layout(
        height=420,
        xaxis=dict(
            range=[
                0,
                max_value,
            ]
        ),
        coloraxis_colorbar=dict(
            title="Risco %",
        ),
    )

    return fig


def grafico_heatmap(df):

    dados = df.copy()

    dias_semana = {
        0: "Segunda-feira",
        1: "Terça-feira",
        2: "Quarta-feira",
        3: "Quinta-feira",
        4: "Sexta-feira",
        5: "Sábado",
        6: "Domingo",
    }

    dados["dia_semana"] = (
        dados["data"]
        .dt.dayofweek
        .map(dias_semana)
    )

    ordem = [
        "Segunda-feira",
        "Terça-feira",
        "Quarta-feira",
        "Quinta-feira",
        "Sexta-feira",
        "Sábado",
        "Domingo",
    ]

    pivot = (
        dados
        .groupby(
            [
                "dia_semana",
                "produto",
            ]
        )["quantidade"]
        .sum()
        .reset_index()
    )

    tabela = (
        pivot.pivot(
            index="dia_semana",
            columns="produto",
            values="quantidade",
        )
        .fillna(0)
    )

    tabela = tabela.reindex(
        ordem
    )

    fig = px.imshow(
        tabela,
        text_auto=True,
        aspect="auto",
        title=(
            "Heatmap de incidentes "
            "por dia da semana"
        ),
        color_continuous_scale="Blues",
    )

    fig.update_layout(
        height=430
    )

    return fig



def gerar_grafico_png(
    funcao,
    *args,
):

    fig, ax = plt.subplots(
        figsize=(10, 4.8)
    )

    funcao(
        *args,
        ax=ax,
    )

    buffer = io.BytesIO()

    plt.tight_layout()

    plt.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer


def chart_pdf_historico(
    df,
    previsao_df,
    produto_foco,
    ax,
):

    if produto_foco != "Todos":

        dados = df[
            df["produto"] == produto_foco
        ]

    else:

        dados = df

    serie = (
        dados
        .groupby("data")["quantidade"]
        .sum()
    )

    ax.plot(
        serie.index,
        serie.values,
        marker="o",
        label="Histórico",
    )

    ax.plot(
        previsao_df["data"],
        previsao_df["previsao"],
        marker="o",
        label="Previsão",
    )

    ax.fill_between(
        previsao_df["data"],
        previsao_df[
            "limite_inferior"
        ],
        previsao_df[
            "limite_superior"
        ],
        alpha=0.20,
        label="Intervalo 95%",
    )

    ax.set_title(
        "Histórico + previsão — "
        f"{produto_foco}"
    )

    ax.set_ylabel(
        "Incidentes"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()



def chart_pdf_risco(
    risco_df,
    ax,
):

    dados = (
        risco_df
        .sort_values(
            "risco_ola"
        )
    )

    ax.barh(
        dados["equipe"],
        dados["risco_ola"],
    )

    ax.set_title(
        "Risco estimado de OLA por equipe"
    )

    ax.set_xlabel(
        "Risco (%)"
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )


def chart_pdf_pareto(
    df,
    ax,
):

    dados = (
        df.groupby("produto")[
            "quantidade"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    ax.bar(
        dados.index,
        dados.values,
    )

    ax.set_title(
        "Incidentes por produto"
    )

    ax.set_ylabel(
        "Quantidade"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )



def gerar_pdf(
    df,
    previsao_df,
    risco_df,
    alertas,
    produto_foco,
):

    buffer_pdf = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer_pdf,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TituloCentroCustom",
            parent=styles["Title"],
            alignment=TA_CENTER,
            spaceAfter=15,
        )
    )

    story = []


    story.append(
        Paragraph(
            "AIOps Locaweb — "
            "Relatório Analítico-Preditivo",
            styles["TituloCentroCustom"],
        )
    )

    story.append(
        Paragraph(
            "Gerado em "
            + datetime.now().strftime(
                "%d/%m/%Y %H:%M"
            ),
            styles["Normal"],
        )
    )

    story.append(
        Spacer(
            1,
            10,
        )
    )

  
    data_max = df["data"].max()

    total = int(
        df["quantidade"].sum()
    )

    ultimo_dia = int(
        df[
            df["data"] == data_max
        ]["quantidade"].sum()
    )

    pred_d1 = int(
        round(
            previsao_df
            .iloc[0]["previsao"]
        )
    )

    pred_d7 = int(
        round(
            previsao_df
            .iloc[-1]["previsao"]
        )
    )

    risco_max = float(
        risco_df[
            "risco_ola"
        ].max()
    )

    kpi_data = [
        [
            "Indicador",
            "Valor",
        ],
        [
            "Incidentes no período",
            f"{total:,}".replace(
                ",",
                ".",
            ),
        ],
        [
            "Incidentes no último dia",
            str(ultimo_dia),
        ],
        [
            "Previsão D+1",
            str(pred_d1),
        ],
        [
            "Previsão D+7",
            str(pred_d7),
        ],
        [
            "Maior risco de OLA",
            f"{risco_max:.1f}%",
        ],
        [
            "Produto analisado",
            produto_foco,
        ],
    ]

    tabela_kpi = Table(
        kpi_data,
        colWidths=[
            8 * cm,
            8 * cm,
        ],
    )

    tabela_kpi.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#1f2937"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        tabela_kpi
    )

    story.append(
        Spacer(
            1,
            15,
        )
    )



    img = gerar_grafico_png(
        chart_pdf_historico,
        df,
        previsao_df,
        produto_foco,
    )

    story.append(
        RLImage(
            img,
            width=17 * cm,
            height=8.2 * cm,
        )
    )

    story.append(
        PageBreak()
    )

    
    img = gerar_grafico_png(
        chart_pdf_risco,
        risco_df,
    )

    story.append(
        RLImage(
            img,
            width=17 * cm,
            height=8 * cm,
        )
    )

    story.append(
        Spacer(
            1,
            15,
        )
    )


    img = gerar_grafico_png(
        chart_pdf_pareto,
        df,
    )

    story.append(
        RLImage(
            img,
            width=17 * cm,
            height=8 * cm,
        )
    )

    story.append(
        PageBreak()
    )


    story.append(
        Paragraph(
            "Alertas e recomendações",
            styles["Heading2"],
        )
    )

    for alerta in alertas:

        texto = (
            f"<b>"
            f"{html.escape(alerta['nivel'])}"
            f" — "
            f"{html.escape(alerta['titulo'])}"
            f"</b><br/>"
            f"{html.escape(alerta['mensagem'])}"
        )

        story.append(
            Paragraph(
                texto,
                styles["BodyText"],
            )
        )

        story.append(
            Spacer(
                1,
                8,
            )
        )


    story.append(
        Paragraph(
            "Equipes com maior risco",
            styles["Heading2"],
        )
    )

    tabela_risco = [
        [
            "Equipe",
            "Risco",
            "Classificação",
        ]
    ]

    for _, row in (
        risco_df
        .head(5)
        .iterrows()
    ):

        tabela_risco.append(
            [
                row["equipe"],
                f"{row['risco_ola']:.1f}%",
                str(
                    row["classificacao"]
                ),
            ]
        )

    tabela = Table(
        tabela_risco,
        colWidths=[
            7 * cm,
            4 * cm,
            5 * cm,
        ],
    )

    tabela.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#1f2937"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        tabela
    )

    doc.build(
        story
    )

    buffer_pdf.seek(0)

    return buffer_pdf.getvalue()



def gerar_html_report(
    df,
    fig_hist,
    fig_pareto,
    fig_prioridade,
    fig_risco,
    alertas,
):

    total = int(
        df["quantidade"].sum()
    )

    html_alertas = ""

    for alerta in alertas:

        html_alertas += f"""
        <div style="
            border-left:5px solid #e25555;
            padding:12px;
            margin-bottom:10px;
            background:#ffffff;
            border-radius:6px;
        ">
            <strong>
                {html.escape(alerta["nivel"])}
                —
                {html.escape(alerta["titulo"])}
            </strong>
            <br>
            {html.escape(alerta["mensagem"])}
        </div>
        """

    return f"""
    <!DOCTYPE html>

    <html lang="pt-BR">

    <head>

        <meta charset="utf-8">

        <title>
            AIOps Locaweb
        </title>

    </head>

    <body style="
        font-family:Arial,sans-serif;
        max-width:1400px;
        margin:auto;
        padding:40px;
        background:#f7f9fc;
    ">

        <h1>
            AIOps Locaweb —
            Relatório Analítico-Preditivo
        </h1>

        <p>
            Gerado em
            {datetime.now().strftime(
                "%d/%m/%Y %H:%M"
            )}
        </p>

        <div style="
            background:white;
            padding:20px;
            border-radius:10px;
            margin-bottom:20px;
        ">

            <h2>
                Resumo Executivo
            </h2>

            <p>
                Total de incidentes analisados:
                <strong>
                    {total}
                </strong>
            </p>

        </div>

        <h2>
            Histórico + previsão
        </h2>

        {fig_hist.to_html(
            full_html=False,
            include_plotlyjs=True,
        )}

        <h2>
            Pareto
        </h2>

        {fig_pareto.to_html(
            full_html=False,
            include_plotlyjs=False,
        )}

        <h2>
            Prioridades
        </h2>

        {fig_prioridade.to_html(
            full_html=False,
            include_plotlyjs=False,
        )}

        <h2>
            Risco de OLA
        </h2>

        {fig_risco.to_html(
            full_html=False,
            include_plotlyjs=False,
        )}

        <h2>
            Alertas
        </h2>

        {html_alertas}

    </body>

    </html>
    """


st.sidebar.header(
    "⚙️ Configuração"
)

modo_dados = st.sidebar.radio(
    "Fonte dos dados",
    [
        "Simulação",
        "Upload CSV",
    ],
)



if modo_dados == "Upload CSV":

    arquivo = (
        st.sidebar.file_uploader(
            "Envie o CSV de incidentes",
            type=["csv"],
            help=(
                "Colunas esperadas: "
                "data, produto, prioridade, "
                "quantidade, equipe"
            ),
        )
    )

    if arquivo is None:

        st.info(
            "Envie um CSV para utilizar "
            "dados reais."
        )

        st.stop()

    try:

        df_hist = normalizar_dados(
            pd.read_csv(
                arquivo
            )
        )

    except Exception as exc:

        st.error(
            "Erro ao processar o arquivo: "
            f"{exc}"
        )

        st.stop()

else:

    df_hist = (
        gerar_dados_simulados()
    )



st.sidebar.divider()

produtos = sorted(
    df_hist["produto"]
    .unique()
)

equipes = sorted(
    df_hist["equipe"]
    .unique()
)

prioridades = sorted(
    df_hist["prioridade"]
    .unique()
)

produto_foco = (
    st.sidebar.selectbox(
        "Produto",
        [
            "Todos"
        ] + produtos,
    )
)

equipes_selecionadas = (
    st.sidebar.multiselect(
        "Equipes",
        equipes,
        default=equipes,
    )
)

prioridades_selecionadas = (
    st.sidebar.multiselect(
        "Prioridades",
        prioridades,
        default=prioridades,
    )
)

max_dias = max(
    14,
    int(
        (
            df_hist["data"].max()
            - df_hist["data"].min()
        ).days
    ),
)

dias_historico = (
    st.sidebar.slider(
        "Janela histórica",
        min_value=14,
        max_value=min(
            120,
            max_dias,
        ),
        value=min(
            60,
            max_dias,
        ),
    )
)



df = df_hist.copy()

df = df[
    df["equipe"]
    .isin(
        equipes_selecionadas
    )
]

df = df[
    df["prioridade"]
    .isin(
        prioridades_selecionadas
    )
]

if produto_foco != "Todos":

    df = df[
        df["produto"]
        == produto_foco
    ]

if df.empty:

    st.warning(
        "Nenhum dado disponível "
        "com os filtros selecionados."
    )

    st.stop()

data_max = df["data"].max()

df = df[
    df["data"]
    >= data_max
    - timedelta(
        days=dias_historico - 1
    )
]

if df.empty:

    st.warning(
        "Não existem dados "
        "para a janela selecionada."
    )

    st.stop()


produto_previsao = (
    produto_foco
    if produto_foco != "Todos"
    else None
)

try:

    (
        serie_prev,
        previsao_df,
        metricas_modelo,
        importancia_modelo,
    ) = prever_volume(
        df,
        produto=produto_previsao,
        dias=7,
    )

except ValueError as exc:

    st.warning(
        str(exc)
    )

    st.stop()


risco_df = (
    calcular_risco_ola(
        df
    )
)


alertas = (
    gerar_alertas(
        df,
        previsao_df,
        risco_df,
        produto_previsao
        or "Todos os produtos",
    )
)


ultima_data = (
    df["data"].max()
)

total_hoje = int(
    df[
        df["data"]
        == ultima_data
    ]["quantidade"]
    .sum()
)

media_7d = float(
    df[
        df["data"]
        > ultima_data
        - timedelta(days=7)
    ]["quantidade"]
    .mean()
)

previsao_d1 = float(
    previsao_df
    .iloc[0]["previsao"]
)

previsao_d7 = float(
    previsao_df
    .iloc[-1]["previsao"]
)

risco_max = float(
    risco_df[
        "risco_ola"
    ].max()
)

variacao_d1 = (
    previsao_d1
    - media_7d
)

percentual_d1 = (
    (
        previsao_d1
        - media_7d
    )
    / media_7d
    * 100
    if media_7d > 0
    else 0
)


st.subheader("🤖 Desempenho do modelo preditivo")
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.metric("MAE", f"{metricas_modelo['MAE']:.2f}", help="Erro absoluto médio na validação temporal.")
with mc2:
    st.metric("RMSE", f"{metricas_modelo['RMSE']:.2f}", help="Penaliza erros maiores.")
with mc3:
    r2v = metricas_modelo["R2"]
    st.metric("R²", "N/D" if np.isnan(r2v) else f"{r2v:.3f}", help="Desempenho no conjunto de teste temporal.")
with mc4:
    st.metric("Limiar de pico", f"{metricas_modelo['limiar_pico']:.1f}")

with st.expander("🔬 Variáveis mais importantes"):
    imp = importancia_modelo.copy()
    imp["importancia"] = (imp["importancia"] * 100).round(2)
    st.dataframe(imp.rename(columns={"variavel":"Variável", "importancia":"Importância (%)"}).head(10),
                 use_container_width=True, hide_index=True)

st.divider()



c1, c2, c3, c4, c5 = (
    st.columns(5)
)

with c1:

    st.metric(
        "📅 Incidentes hoje",
        f"{total_hoje:,}".replace(
            ",",
            ".",
        ),
    )

with c2:

    st.metric(
        "🔮 Previsão D+1",
        f"{previsao_d1:.0f}",
        f"{variacao_d1:+.0f} vs média",
    )

with c3:

    st.metric(
        "📆 Previsão D+7",
        f"{previsao_d7:.0f}",
    )

with c4:

    st.metric(
        "📈 Variação D+1",
        f"{percentual_d1:+.1f}%",
    )

with c5:

    st.metric(
        "⚠️ Maior risco OLA",
        f"{risco_max:.1f}%",
    )



st.subheader(
    "📈 Evolução e previsão"
)

fig_hist = (
    grafico_historico_previsao(
        df,
        previsao_df,
        produto_foco
        or "Todos",
    )
)

st.plotly_chart(
    fig_hist,
    use_container_width=True,
)

st.subheader("🔮 Previsão diária de incidentes")
tabela_previsao = previsao_df[["data", "previsao", "limite_inferior", "limite_superior",
                              "risco_pico", "classificacao_risco"]].copy()
tabela_previsao["data"] = tabela_previsao["data"].dt.strftime("%d/%m/%Y")
tabela_previsao = tabela_previsao.rename(columns={
    "data":"Data", "previsao":"Incidentes previstos", "limite_inferior":"Limite inferior",
    "limite_superior":"Limite superior", "risco_pico":"Risco de pico (%)",
    "classificacao_risco":"Risco"})
st.dataframe(tabela_previsao.round(1), use_container_width=True, hide_index=True)



col_a, col_b = (
    st.columns(2)
)

with col_a:

    st.subheader(
        "🎯 Pareto"
    )

    fig_pareto = (
        grafico_pareto(
            df
        )
    )

    st.plotly_chart(
        fig_pareto,
        use_container_width=True,
    )

with col_b:

    st.subheader(
        "🚨 Prioridades"
    )

    fig_prioridade = (
        grafico_prioridades(
            df
        )
    )

    st.plotly_chart(
        fig_prioridade,
        use_container_width=True,
    )



st.subheader(
    "🗓️ Heatmap operacional"
)

fig_heatmap = (
    grafico_heatmap(
        df
    )
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True,
)



st.subheader(
    "⚠️ Risco estimado de OLA"
)

fig_risco = (
    grafico_risco_ola(
        risco_df
    )
)

st.plotly_chart(
    fig_risco,
    use_container_width=True,
)



tabela_risco = (
    risco_df[
        [
            "equipe",
            "recente",
            "anterior",
            "variacao",
            "risco_ola",
            "classificacao",
        ]
    ]
    .rename(
        columns={
            "equipe":
                "Equipe",

            "recente":
                "Volume ponderado 7d",

            "anterior":
                "Volume ponderado período anterior",

            "variacao":
                "Variação %",

            "risco_ola":
                "Risco OLA %",

            "classificacao":
                "Classificação",
        }
    )
)

st.dataframe(
    tabela_risco,
    use_container_width=True,
    hide_index=True,
)



st.subheader(
    "🔔 Alertas acionáveis"
)

for alerta in alertas:

    nivel = (
        alerta["nivel"]
    )

    titulo = (
        alerta["titulo"]
    )

    mensagem = (
        alerta["mensagem"]
    )

    if nivel == "Crítico":

        st.error(
            f"**{titulo}** — "
            f"{mensagem}"
        )

    elif nivel == "Atenção":

        st.warning(
            f"**{titulo}** — "
            f"{mensagem}"
        )

    else:

        st.success(
            f"**{titulo}** — "
            f"{mensagem}"
        )



with st.expander(
    "🔎 Ver dados tratados"
):

    st.dataframe(
        df.sort_values(
            "data",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )



st.subheader(
    "📥 Exportações"
)

col_d1, col_d2, col_d3 = (
    st.columns(3)
)



csv_bytes = (
    df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )
)

with col_d1:

    st.download_button(
        label="⬇️ Baixar dados CSV",
        data=csv_bytes,
        file_name=(
            "aiops_incidentes.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )



html_report = (
    gerar_html_report(
        df,
        fig_hist,
        fig_pareto,
        fig_prioridade,
        fig_risco,
        alertas,
    )
)

with col_d2:

    st.download_button(
        label="🌐 Baixar relatório HTML",
        data=html_report,
        file_name=(
            "relatorio_aiops.html"
        ),
        mime="text/html",
        use_container_width=True,
    )



with col_d3:

    try:

        pdf_report = (
            gerar_pdf(
                df,
                previsao_df,
                risco_df,
                alertas,
                produto_foco
                or "Todos os produtos",
            )
        )

        st.download_button(
            label="📄 Baixar relatório PDF",
            data=pdf_report,
            file_name=(
                "relatorio_aiops.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as exc:

        st.error(
            "Não foi possível gerar "
            "o relatório PDF."
        )

        st.caption(
            str(exc)
        )



st.divider()

st.markdown(
    """
    <div class="footer">

    Protótipo AIOps para demonstração.

    O risco de OLA é estimado a partir de volume de incidentes
    ponderado por prioridade e evolução recente.

    Em produção, recomenda-se integrar:
    ITSM, SLA/OLA, backlog, MTTR, MTBF,
    monitoramento, logs, eventos, deploys,
    mudanças e modelos preditivos.

    </div>
    """,
    unsafe_allow_html=True,
)
