import io
import pandas as pd

from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi import HTTPException 
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
@router.post("/upload")
async def upload_dashboard_csv(file: UploadFile = File(...)):
    target = DATA_DIR / "dashboard_uploaded.csv"

    contents = await file.read()
    target.write_bytes(contents)

    global _DASHBOARD_CSV_PATH
    _DASHBOARD_CSV_PATH = target

    return RedirectResponse(url="/dashboard", status_code=303)

def _load_dashboard_df() -> pd.DataFrame:
    if _DASHBOARD_CSV_PATH is None:
        raise HTTPException(400, "CSV ещё не загружен. Сначала POST /dashboard/upload")

    return pd.read_csv(_DASHBOARD_CSV_PATH, encoding="utf-8")

def _clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise HTTPException(400, f"Нет колонки '{col}'. Колонки: {list(df.columns)}")

    s = df[col].astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan") & (s.str.lower() != "undefined")]
    return s

templates = Jinja2Templates(directory="templates")

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "exported_s7_data.csv"
_DASHBOARD_CSV_PATH: Path 

@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/topic-pie.png") 
def topic_pie_png():
    df = _load_dashboard_df()

    topic_counts = _clean_series(df, "user_topic").value_counts()
    total = topic_counts.sum()
    topic_percentages = (topic_counts / total * 100).round(2)

    colors = plt.cm.tab20c(range(len(topic_percentages)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.pie(topic_percentages.values, labels=None, autopct="", colors=colors, startangle=140)
    ax.axis("equal")
    plt.title("Распределение тематик обращений (%)", fontsize=14, pad=20)

    legend_labels = [
        f"{topic} - {percentage}%"
        for topic, percentage in zip(topic_percentages.index, topic_percentages.values)
    ]
    legend_patches = [mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
    plt.legend(handles=legend_patches, title="Тематики", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@router.get("/sentiment-pie.png") 
def sentiment_pie_png():
    df = _load_dashboard_df()
    
    sentiment_counts = _clean_series(df, "user_sentiment").value_counts()
    total = sentiment_counts.sum()
    sentiment_percentages = (sentiment_counts / total * 100).round(2)
    
    colors = plt.cm.tab20c(range(len(sentiment_percentages)))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(
        sentiment_percentages.values,
        labels=None,
        autopct="",
        colors=colors,
        startangle=140
    )
    ax.axis("equal")
    plt.title("Распределение сентимента", fontsize=20, pad=20)

    legend_patches = []
    for i in range(len(sentiment_percentages)):
        legend_patches.append(
            mpatches.Patch(
                color=colors[i],
                label=f"{sentiment_percentages.index[i]} - {sentiment_percentages.values[i]}%"
            )
        )

    plt.legend(
        handles=legend_patches,
        title="Сентимент",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=15
    )

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@router.get("/emotion-bubble.png") 
def emotion_bubble_png():
    df = _load_dashboard_df()
    
    emotion_counts = _clean_series(df, "user_emotion").value_counts()

    if len(df) == 0:
        emotion_percentages = pd.Series([100.0], index=["unknown"])
    else:
        emotion_percentages = (emotion_counts / len(df) * 100).round(1)

    emotions = emotion_percentages.index.tolist()
    percentages = emotion_percentages.values.tolist()

    np.random.seed(5)  # воспроизводимость
    x = np.random.rand(len(emotions)) * 10
    y = np.random.rand(len(emotions)) * 10

    sizes = [p * 500 for p in percentages]
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotions))) if len(emotions) > 0 else ["grey"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 8),
        gridspec_kw={"width_ratios": [2, 1]}
    )

    ax1.scatter(x, y, s=sizes, c=colors, alpha=0.7,
                edgecolors="black", linewidth=1.5)

    ax1.set_title("Распределение эмоций в клиентских сообщениях", fontsize=16, fontweight="regular", pad=20)
    ax1.set_xlabel("Координата X", fontsize=12)
    ax1.set_ylabel("Координата Y", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    for i, emotion in enumerate(emotions):
        ax1.text(x[i], y[i], emotion, fontsize=10, fontweight="regular",
                 ha="center", va="center", color="black")

    legend_elements = []
    for i, (emotion, percentage) in enumerate(zip(emotions, percentages)):
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1,
                          fc=colors[i],
                          label=f"{emotion}: {percentage}%",
                          edgecolor="black")
        )

    ax2.legend(handles=legend_elements, loc="center", fontsize=12,
               title="Эмоции", title_fontsize=14)
    ax2.axis("off")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

@router.get("/emotion-topic", response_class=HTMLResponse)
async def emotion_topic_dashboard(request: Request):
    global _DASHBOARD_CSV_PATH

    if not _DASHBOARD_CSV_PATH or not os.path.exists(_DASHBOARD_CSV_PATH):
        return HTMLResponse(
            "<h3>CSV не загружен</h3><p>Сначала загрузите файл на странице /dashboard.</p>",
            status_code=400
        )

    import plotly.graph_objects as go

    df = pd.read_csv(_DASHBOARD_CSV_PATH)

    df_clean = df[~df["user_emotion"].isin(["undefined", "invalid"])]
    df_clean = df_clean.dropna(subset=["user_emotion"])

    emotion_counts = df_clean["user_emotion"].value_counts()
    emotion_percentages = (emotion_counts / emotion_counts.sum() * 100).round(2)

    emotion_topics = {}
    for emotion in df_clean["user_emotion"].unique():
        topics = (
            df_clean[df_clean["user_emotion"] == emotion]["user_topic"]
            .dropna()
            .unique()
        )
        emotion_topics[emotion] = [str(topic).strip() for topic in topics]

    emotions = emotion_percentages.index.tolist()
    percentages = emotion_percentages.values.tolist()

    colors = ["#4285F4", "#FBBC05", "#EA4335", "#34A853"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=emotions,
                values=percentages,
                marker=dict(colors=colors),
                textinfo="percent",
                hovertemplate=(
                    "<b>Эмоция:</b> %{label}<br>"
                    "<b>Доля:</b> %{percent}<br>"
                    "<b>Тематики:</b><br>%{customdata}"
                    "<extra></extra>"
                ),
                customdata=[
                    "<br>".join(emotion_topics[e]) for e in emotions
                ],
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>Распределение эмоций",
            x=0.5,
            xanchor="center",
            font=dict(size=25, family="Arial"),
        ),
        width=1200,
        height=800,
    )

    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    return HTMLResponse(html)