from fastapi import FastAPI
from app.routes import auth_router
from app.routes import med_router
from app.routes import user_router
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

app.include_router(med_router.router, prefix="/api")
app.include_router(auth_router.router, prefix="/api")
app.include_router(user_router.router, prefix="/api")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)