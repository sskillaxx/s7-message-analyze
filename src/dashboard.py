import io
import os
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import pyarrow

import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
templates = Jinja2Templates(directory="templates")

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "exported_s7_data.csv"

def _load_dashboard_df() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise HTTPException(400, "Файл exported_s7_data.csv не найден. Сначала выполните /dataset/upload-dataset-export")

    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(CSV_PATH, encoding="cp1251")
    
    return df

def _clean_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise HTTPException(400, f"Нет колонки '{col}'. Колонки: {list(df.columns)}")

    s = df[col].astype(str).str.strip()
    s = s[(s != "") & (s.str.lower() != "nan") & (s.str.lower() != "undefined")]
    return s

@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/get-date-range")
def get_date_range():
    df = _load_dashboard_df()
    
    if "message_time" not in df.columns:
        raise HTTPException(400, "Колонка 'message_time' не найдена в файле exported_s7_data.csv")
    
    # Преобразуем колонку в datetime
    date_col = pd.to_datetime(df["message_time"], errors='coerce')
    
    # Удаляем NaT значения
    date_col = date_col.dropna()
    
    if date_col.empty:
        raise HTTPException(400, f"Колонка 'message_time' не содержит корректных дат")
    
    min_date = date_col.min().strftime('%Y-%m-%d')
    max_date = date_col.max().strftime('%Y-%m-%d')
    
    return {"min_date": min_date, "max_date": max_date}


@router.get("/topic-pie.png")
def topic_pie_png(
    start_date: str | None = Query(default=None, description="Начало периода: YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Конец периода: YYYY-MM-DD"),
):
    df = _load_dashboard_df()

    start = _parse_date_param(start_date, is_end=False)
    end = _parse_date_param(end_date, is_end=True)
    df = _filter_df_by_message_time(df, start, end)

    topic_counts = _clean_series(df, "user_topic").value_counts()
    if topic_counts.empty:
        # Добавим отладочную информацию
        total_rows = len(df)
        non_empty_topics = df['user_topic'].dropna().astype(str).str.strip().nunique()
        raise HTTPException(
            400,
            f"Нет данных для построения графика тематик. "
            f"Всего строк: {total_rows}, уникальных непустых тем: {non_empty_topics}. "
            f"Проверьте наличие колонки 'user_topic' и её содержимое."
        )

    total = topic_counts.sum()
    topic_percentages = (topic_counts / total * 100).round(2)

    colors = plt.cm.tab20c(range(len(topic_percentages)))

    fig, ax = plt.subplots(figsize=(12, 6))
    wedges, texts, autotexts = ax.pie(
        topic_percentages.values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.85,
    )

    for autotext in autotexts:
        autotext.set_color("black")
        autotext.set_fontweight("regular")
        autotext.set_fontsize(7)

    ax.axis("equal")
    plt.title("Распределение тематик обращений (%)", fontsize=14, pad=20)

    legend_labels = [
        f"{topic} - {percentage}%"
        for topic, percentage in zip(topic_percentages.index, topic_percentages.values)
    ]
    legend_patches = [
        mpatches.Patch(color=colors[i], label=legend_labels[i])
        for i in range(len(legend_labels))
    ]

    plt.legend(
        handles=legend_patches,
        title="Тематики",
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
        fontsize=10,
    )

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@router.get("/sentiment-pie.png")
def sentiment_pie_png(
    start_date: str | None = Query(default=None, description="Начало периода: YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Конец периода: YYYY-MM-DD"),
):
    df = _load_dashboard_df()

    start = _parse_date_param(start_date, is_end=False)
    end = _parse_date_param(end_date, is_end=True)
    df = _filter_df_by_message_time(df, start, end)

    sentiment_counts = _clean_series(df, "user_sentiment").value_counts()

    NUM_sentiment_TO_SHOW = 3
    top_sentiments = sentiment_counts.head(NUM_sentiment_TO_SHOW)
    if top_sentiments.empty:
        total_rows = len(df)
        non_empty_sentiments = df['user_sentiment'].dropna().astype(str).str.strip().nunique()
        raise HTTPException(
            400,
            f"Нет данных для построения графика сентимента. "
            f"Всего строк: {total_rows}, уникальных непустых сентиментов: {non_empty_sentiments}. "
            f"Проверьте наличие колонки 'user_sentiment' и её содержимое."
        )

    total = top_sentiments.sum()
    sentiment_percentages = (top_sentiments / total * 100).round(2)

    n = len(sentiment_percentages)
    colors = plt.cm.tab20c(range(n))

    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(
        sentiment_percentages.values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.85,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    ax.axis("equal")
    plt.title("Распределение сентимента", fontsize=30, pad=20)

    legend_patches = []
    for i in range(n):
        sentiment = sentiment_percentages.index[i]
        percentage = sentiment_percentages.values[i]
        legend_patches.append(
            mpatches.Patch(color=colors[i], label=f"{sentiment} - {percentage}%")
        )

    plt.legend(
        handles=legend_patches,
        title="Сентимент",
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
        fontsize=18,
    )

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@router.get("/emotion-bubble.png")
def emotion_bubble_png(
    start_date: str | None = Query(default=None, description="Начало периода: YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Конец периода: YYYY-MM-DD"),
):
    df = _load_dashboard_df()

    start = _parse_date_param(start_date, is_end=False)
    end = _parse_date_param(end_date, is_end=True)
    df = _filter_df_by_message_time(df, start, end)

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
async def emotion_topic_dashboard(
    request: Request,
    start_date: str | None = Query(default=None, description="Начало периода: YYYY-MM-DD"),
    end_date: str | None = Query(default=None, description="Конец периода: YYYY-MM-DD"),
):
    # Используем тот же источник данных, что и остальные дашборды
    df = _load_dashboard_df()

    

    start = _parse_date_param(start_date, is_end=False)
    end = _parse_date_param(end_date, is_end=True)
    df = _filter_df_by_message_time(df, start, end)

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


def _parse_date_param(value: str | None, *, is_end: bool) -> datetime | None:
    """
    Поддерживаем:
    - YYYY-MM-DD (из <input type="date">)

    Для start используем начало дня, для end — конец дня (включительно).
    """
    if value is None or str(value).strip() == "":
        return None

    v = str(value).strip()

    # YYYY-MM-DD
    try:
        d = date.fromisoformat(v)
        return datetime.combine(d, time.max if is_end else time.min)
    except ValueError:
        raise HTTPException(
            400,
            "Некорректный формат даты. Ожидаю YYYY-MM-DD.",
        )


def _filter_df_by_message_time(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """Фильтрует df по колонке message_time в интервале [start, end]."""
    if start is None and end is None:
        return df

    if "message_time" not in df.columns:
        raise HTTPException(400, "Для фильтрации по периоду нужна колонка 'message_time' в CSV.")

    ts = pd.to_datetime(df["message_time"], errors="coerce")
    if ts.isna().all():
        raise HTTPException(400, f"Колонка 'message_time' не распознана как даты/время (все значения NaT).")

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts <= end

    return df.loc[mask].copy()