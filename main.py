from fastapi import FastAPI

from src.api import api_router
from src.dashboard import router as dashboard_router

app = FastAPI(
    title="API FOR S7_CASE_3",
    description=(
        '''
        локальный API: 
        1. загрузка датасета
        2. определение в каждом элементе датасета языка, сентимента, эмоции, тематики
        3. запись результатов в БД с последующим экспортом в .csv
        '''
    ),
    version="0.1.0",
)

app.include_router(api_router)
app.include_router(dashboard_router)

@app.get("/")
def root():
    return {
        "message": "api works",
        "docs": "/docs",
        "openapi_json": "/openapi.json",
    }

def start_fastapi() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    start_fastapi()