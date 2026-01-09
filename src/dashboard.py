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

def _format_date_range_for_title(start: datetime | None, end: datetime | None) -> str:
    """
    Формирует строку с диапазоном дат для заголовка графика в формате DD.MM.YYYY.
    Если обе даты заданы, возвращает "\nПериод: DD.MM.YYYY – DD.MM.YYYY".
    Если одна из дат не задана, возвращает пустую строку.
    """
    if start and end:
        start_fmt = start.strftime('%d.%m.%Y')  # DD.MM.YYYY
        end_fmt = end.strftime('%d.%m.%Y')      # DD.MM.YYYY
        return f"\nПериод: {start_fmt} – {end_fmt}"
    return ""

@router.get("/", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/get-date-range")
def get_date_range():
    df = _load_dashboard_df()
    
    if "message_time" not in df.columns:
        raise HTTPException(400, "Колонка 'message_time' не найдена в файле exported_s7_data.csv")
    
    # Явно указываем формат даты из CSV: DD.MM.YYYY HH:MM
    date_col = pd.to_datetime(df["message_time"], format='%d.%m.%Y %H:%M', errors='coerce')
    
    # Удаляем NaT значения
    date_col = date_col.dropna()
    
    if date_col.empty:
        raise HTTPException(400, f"Колонка 'message_time' не содержит корректных дат")
    
    min_date = date_col.min().strftime('%d.%m.%Y')  # DD.MM.YYYY
    max_date = date_col.max().strftime('%d.%m.%Y')  # DD.MM.YYYY
    
    return {"min_date": min_date, "max_date": max_date}


@router.get("/topic-pie.png")
def topic_pie_png(
    start_date: str | None = Query(default=None, description="Начало периода: DD.MM.YYYY"),
    end_date: str | None = Query(default=None, description="Конец периода: DD.MM.YYYY"),
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
    
    date_range_str = _format_date_range_for_title(start, end)
    title = f"Распределение тематик обращений (%)"
    if date_range_str:
        title += date_range_str
        
    plt.title(title, fontsize=14, pad=20)

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
    start_date: str | None = Query(default=None, description="Начало периода: DD.MM.YYYY"),
    end_date: str | None = Query(default=None, description="Конец периода: DD.MM.YYYY"),
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
    
    date_range_str = _format_date_range_for_title(start, end)
    title = "Распределение сентимента"
    if date_range_str:
        title += date_range_str
    
    plt.title(title, fontsize=30, pad=20)

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


@router.get("/emotion-topic", response_class=HTMLResponse)
async def emotion_topic_dashboard(
    request: Request,
    start_date: str | None = Query(default=None, description="Начало периода: DD.MM.YYYY"),
    end_date: str | None = Query(default=None, description="Конец периода: DD.MM.YYYY"),
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

    # --- Формируем заголовок для Plotly ---
    date_range_str = _format_date_range_for_title(start, end)
    plotly_title = "<b>Распределение эмоций</b>"
    if date_range_str:
        # Убираем символ перевода строки из строки для Plotly, заменяем на пробел
        clean_date_str = date_range_str.replace('\n', ' ')
        plotly_title += f"<br><sub>{clean_date_str}</sub>"

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
            text=plotly_title,
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
    - DD.MM.YYYY (русский формат)

    Для start используем начало дня, для end — конец дня (включительно).
    """
    if value is None or str(value).strip() == "":
        return None

    v = str(value).strip()

    # DD.MM.YYYY
    try:
        d = datetime.strptime(v, "%d.%m.%Y").date()
        return datetime.combine(d, time.max if is_end else time.min)
    except ValueError:
        raise HTTPException(
            400,
            "Некорректный формат даты. Ожидаю DD.MM.YYYY.",
        )


def _filter_df_by_message_time(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    """Фильтрует df по колонке message_time в интервале [start, end]."""
    if start is None and end is None:
        return df

    if "message_time" not in df.columns:
        raise HTTPException(400, "Для фильтрации по периоду нужна колонка 'message_time' в CSV.")

    # Явно указываем формат даты из CSV: DD.MM.YYYY HH:MM
    ts = pd.to_datetime(df["message_time"], format='%d.%m.%Y %H:%M', errors="coerce")

    if ts.isna().all():
        raise HTTPException(400, f"Колонка 'message_time' не распознана как даты/время. Проверьте формат: DD.MM.YYYY HH:MM")

    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts <= end

    return df.loc[mask].copy()