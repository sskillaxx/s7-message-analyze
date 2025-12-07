# src/services.py
from pathlib import Path
import shutil

from fastapi import UploadFile, HTTPException

from src.processing import database_update

# src/services.py -> src -> корень проекта
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx", ".json", ".parquet"}


async def process_dataset_file(file: UploadFile, limit: int | None) -> str:
    """
    1. Сохраняем загруженный файл в data/uploaded_dataset.<ext>.
    2. Вызываем database_update(<путь>, <limit>).
    3. Получаем путь к CSV, экспортированному из БД, и возвращаем его в API.
    """

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Неподдерживаемое расширение: {suffix}. "
                f"Допустимые: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    saved_path = DATA_DIR / f"uploaded_dataset{suffix}"

    # 1. Сохраняем файл
    try:
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось сохранить файл: {e}",
        )

    # 2. Запускаем обработку + запись в БД + экспорт CSV
    try:
        # ВАЖНО: передаём limit позиционным аргументом
        csv_path = database_update(str(saved_path), limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке датасета и экспорте CSV: {e}",
        )

    return csv_path
