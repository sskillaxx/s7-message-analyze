import io
import pandas as pd

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import matplotlib
matplotlib.use("Agg")  # важно для сервера без GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

router = APIRouter()
templates = Jinja2Templates(directory="templates")

CSV_PATH = "data/exported_s7_data.csv"  # ✅ путь к твоему csv (как в проекте)

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/dashboard/topic-pie.png")
def topic_pie_png():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    # ✅ поддержка обоих вариантов названий колонки
    topic_col = "topic" if "topic" in df.columns else "user_topic"
    if topic_col not in df.columns:
        # если нет вообще — лучше вернуть 400/сообщение, но пока упростим
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