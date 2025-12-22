import os
import csv
import time
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

import pandas as pd
import psycopg2 as ps

from src.nlp import (
    detect_language,
    detect_sentiment,
    detect_emotion,
    detect_topic,
)

def check_file_extension(path: str) -> pd.DataFrame:
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
    print(f"[database_update] Старт обработки файла: {messages_path}")

    df = check_file_extension(messages_path)
    print(f"[database_update] Прочитано строк: {len(df)}, колонки: {list(df.columns)}")

    if limit is not None:
        df = df.head(limit)
        print(f"[database_update] Применён limit={limit}, строк осталось: {len(df)}")

    if "text" not in df.columns:
        raise ValueError(
            f"В датасете нет колонки 'text'. Найденные колонки: {list(df.columns)}"
        )

    has_source = "source" in df.columns
    if has_source:
        print("[database_update] Найдена колонка 'source' — будет использовано условие source == 'CLIENT'.")
    else:
        print("[database_update] Колонки 'source' нет — вставляем ВСЕ строки без фильтра.")

    connection = ps.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        port=os.getenv("DB_PORT"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        client_encoding="UTF8",
    )
    connection.autocommit = True

    cursor = connection.cursor()

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
    db_write_start = None


    for i in range(len(df)):
        text = str(df.iloc[i]["text"]).strip()

        if not text:
            continue

        if has_source:
            source = str(df.iloc[i]["source"])
            if source != "CLIENT":
                continue

        detected_language = detect_language(text)
        detected_sentiment = detect_sentiment(text, language=detected_language)
        detected_emotion = detect_emotion(text, language=detected_language)
        detected_topic = detect_topic(text, language=detected_language)

        if detected_language == "undefined":
            pass
        else:
            data_to_insert = (
            text,
            detected_sentiment,
            detected_emotion,
            detected_topic,
            detected_language,
        )
        
            if db_write_start is None:
                db_write_start = time.perf_counter()

            cursor.execute(insert_query, data_to_insert)
            rows_inserted += 1

    print(f"[database_update] Вставлено НОВЫХ строк в БД: {rows_inserted}")

    db_write_end = time.perf_counter()

    if db_write_start is not None:
        write_time = db_write_end - db_write_start
        print(
            f"[database_update] Время работы: {write_time:.4f} сек "
            f"(строк: {rows_inserted})"
        )

    connection.commit()
    print("[database_update] Транзакция зафиксирована (COMMIT)")

    export_cursor = connection.cursor()
    export_cursor.execute("SELECT * FROM s7_data")
    rows = export_cursor.fetchall()
    col_names = [desc[0] for desc in export_cursor.description]

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    export_path = data_dir / "exported_s7_data.csv"

    with export_path.open("w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(col_names) 
        csv_writer.writerows(rows)

    print(f"[database_update] Экспортировано в CSV: {export_path}")

    export_cursor.close()
    cursor.close()
    connection.close()

    return str(export_path)