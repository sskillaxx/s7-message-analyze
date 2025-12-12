import io
import pandas as pd

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

router = APIRouter()
templates = Jinja2Templates(directory="templates")

CSV_PATH = "data/exported_s7_data.csv"

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/dashboard/topic-pie.png")
def topic_pie_png():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    topic_col = "topic" if "topic" in df.columns else "user_topic"
    if topic_col not in df.columns:
        df[topic_col] = "unknown"

    topic_counts = df[topic_col].value_counts()
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

@router.get("/dashboard/sentiment-pie.png")
def sentiment_pie_png():
    df = pd.read_csv("data/exported_s7_data.csv", encoding="utf-8")

    if "sentiment" not in df.columns:
        df["sentiment"] = "unknown"

    sentiment_counts = df["sentiment"].value_counts()

    NUM_SENTIMENTS_TO_SHOW = 3  # ← прямо тут, без глобальных констант

    top_sentiments = sentiment_counts.head(NUM_SENTIMENTS_TO_SHOW)
    total = top_sentiments.sum()

    if total == 0:
        top_sentiments = pd.Series([1], index=["unknown"])
        total = 1

    sentiment_percentages = (top_sentiments / total * 100).round(2)
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
