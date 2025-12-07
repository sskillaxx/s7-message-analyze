import os
import csv
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

import pandas as pd
import psycopg2 as ps

# когда появятся реальные функции, раскомментируешь и подставишь
# from src.nlp import (
#     detect_language,
#     detect_sentiment,
#     detect_emotion,
#     detect_topic,
# )


def check_file_extension(path: str) -> pd.DataFrame:
    """
    Читаем датасет в pandas.DataFrame по расширению файла.
    """
    extension = os.path.splitext(path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(path)
    elif extension in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    elif extension == ".parquet":
        return pd.read_parquet(path)
    elif extension == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {extension}")


def database_update(messages_path: str, limit: Optional[int] = None) -> str:
    """
    1. Читаем датасет из файла messages_path.
    2. Берём колонку 'text' и (опционально) 'source'.
    3. Цикл for i in range(len(df)):
         - text = df.iloc[i]['text']
         - source = df.iloc[i]['source'] (если колонка есть)
         - если source == 'CLIENT' (или колонки нет — вставляем всё)
           -> вставляем в БД строку с плейсхолдерами:
              detected_language = 'language'
              detected_sentiment = 'sentiment'
              detected_emotion = 'emotion'
              detected_topic = 'topic'
    4. SELECT * FROM s7_data → экспорт в CSV ./data/exported_s7_data.csv.
    5. Возвращаем путь к CSV.
    """

    print(f"[database_update] Старт обработки файла: {messages_path}")

    # === 1. Читаем датасет ===
    df = check_file_extension(messages_path)
    print(f"[database_update] Прочитано строк: {len(df)}, колонки: {list(df.columns)}")

    # === 2. Ограничиваем количество строк (если нужно) ===
    if limit is not None:
        df = df.head(limit)
        print(f"[database_update] Применён limit={limit}, строк осталось: {len(df)}")

    # === 3. Проверяем, что есть колонка 'text' ===
    if "text" not in df.columns:
        raise ValueError(
            f"В датасете нет колонки 'text'. Найденные колонки: {list(df.columns)}"
        )

    has_source = "source" in df.columns
    if has_source:
        print("[database_update] Найдена колонка 'source' — будет использовано условие source == 'CLIENT'.")
    else:
        print("[database_update] Колонки 'source' нет — вставляем ВСЕ строки без фильтра.")

    # === 4. Подключаемся к БД ===
    # ОБЯЗАТЕЛЬНО поменяй параметры под себя
    connection = ps.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        port=os.getenv("DB_PORT"),
        password=os.getenv("DB_PASSWORD"),  # <--- твой пароль
        database=os.getenv("DB_NAME"),         # <--- твоя БД
        client_encoding="UTF8",
    )
    # Чтобы точно не забыть commit — включим autocommit
    connection.autocommit = True

    cursor = connection.cursor()

    # На всякий случай — создаём таблицу, если её нет
    create_table_query = """
        CREATE TABLE IF NOT EXISTS s7_data (
            user_text      TEXT,
            user_sentiment TEXT,
            user_emotion   TEXT,
            user_topic     TEXT,
            user_language  TEXT
        );
    """
    cursor.execute(create_table_query)

    insert_query = """
        INSERT INTO s7_data (user_text, user_sentiment, user_emotion, user_topic, user_language)
        VALUES (%s, %s, %s, %s, %s);
    """

    rows_inserted = 0

    # === 5. ТВОЙ ЦИКЛ: for i in range(len(df)) + условие source == 'CLIENT' ===
    for i in range(len(df)):
        text = str(df.iloc[i]["text"]).strip()

        if not text:
            continue

        if has_source:
            source = str(df.iloc[i]["source"])
            # ТВОЁ УСЛОВИЕ — как есть
            if source != "CLIENT":
                continue

        # --------- СЕЙЧАС: ПЛЕЙСХОЛДЕРЫ ---------
        detected_language = "language"
        detected_sentiment = "sentiment"
        detected_emotion = "emotion"
        detected_topic = "topic"

        # --------- ПОТОМ: ЗАМЕНИШЬ НА РЕАЛЬНЫЕ ФУНКЦИИ ---------
        # detected_language = detect_language(text)
        # detected_sentiment = detect_sentiment(text)
        # detected_emotion = detect_emotion(text)
        # detected_topic = detect_topic(text)
        # -------------------------------------------------------

        data_to_insert = (
            text,
            detected_sentiment,
            detected_emotion,
            detected_topic,
            detected_language,
        )

        cursor.execute(insert_query, data_to_insert)
        rows_inserted += 1

    print(f"[database_update] Вставлено НОВЫХ строк в БД: {rows_inserted}")

    # autocommit=True уже фиксирует изменения, но на всякий случай:
    connection.commit()
    print("[database_update] Транзакция зафиксирована (COMMIT)")

    # === 6. Экспортируем ВСЮ таблицу s7_data в CSV ===

    export_cursor = connection.cursor()
    export_cursor.execute("SELECT * FROM s7_data")
    rows = export_cursor.fetchall()
    col_names = [desc[0] for desc in export_cursor.description]

    # CSV в ./data/exported_s7_data.csv
    base_dir = Path(__file__).resolve().parent.parent  # src -> корень проекта
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    export_path = data_dir / "exported_s7_data.csv"

    with export_path.open("w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(col_names)  # заголовки
        csv_writer.writerows(rows)      # данные

    print(f"[database_update] Экспортировано в CSV: {export_path}")

    export_cursor.close()
    cursor.close()
    connection.close()

    return str(export_path)