from fastapi import FastAPI
from app.routes import (auth, checkprocess, diagnose, legithostpital, location, med, paper, tag, uv)
import os
from dotenv import load_dotenv
import uvicorn
from app.db.mongo import ping_db




load_dotenv()




app = FastAPI()

@app.on_event("startup")
async def startup():
    await ping_db()  # Kiểm tra kết nối MongoDB

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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)