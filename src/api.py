# src/api.py
from pathlib import Path

from fastapi.routing import APIRouter
from fastapi import UploadFile, File, Query
from fastapi.responses import FileResponse

from src.services import process_dataset_file

api_router = APIRouter(
    prefix="/api",
    tags=["dataset"],
)

@api_router.post(
    "/upload-dataset-export",
    summary="загрузка датасета, обновление БД, выгрузка итогового датасета из БД в виде .csv",
    response_description=".csv-файл, экспортированный из БД",
)
async def upload_dataset_and_export(
    file: UploadFile = File(
        ...,
        description="Файл с датасетом (обязательно колонка text; колонка source опциональна)",
    ),
    limit: int | None = Query(
        default=None,
        description="максимальное число строк для обработки",
    ),
) -> FileResponse:
    csv_path = await process_dataset_file(file=file, limit=limit)

    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=Path(csv_path).name,
    )
