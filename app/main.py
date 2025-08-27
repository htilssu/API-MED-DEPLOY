from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.routes import (auth, checkprocess, diagnose, legithostpital, location, med, paper, tag, uv)
import uvicorn
from app.db.mongo import ping_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ping_db()

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    print("App is running")
    return {"message": "Hello World from Thành"}


@app.get("/health")
async def health_check():
    return {"status": "OK"}


app.include_router(auth.router, prefix="/api")
app.include_router(uv.router, prefix="/api")
app.include_router(diagnose.router, prefix="/api")
app.include_router(checkprocess.router, prefix="/api")
app.include_router(legithostpital.router, prefix="/api")
app.include_router(location.router, prefix="/api")
app.include_router(paper.router, prefix="/api")
app.include_router(tag.router, prefix="/api")
app.include_router(med.router, prefix="/api")

if __name__ == "__main__":
    from app.config.setting import setting
    uvicorn.run("app.main:app", host="0.0.0.0", port=setting.PORT)
