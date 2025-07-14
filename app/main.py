from fastapi import FastAPI
from app.routes import (auth_router,checkprocess_router, med_router, uv_router,diagnose_router,legithostpital_router,location_router,paper_router,tag_router)
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

app.include_router(med_router.router, prefix="/med")
app.include_router(auth_router.router, prefix="/auth")
app.include_router(uv_router.router, prefix="/uv")
app.include_router(diagnose_router.router, prefix="/diagnose")
app.include_router(checkprocess_router.router, prefix="/check-process")
app.include_router(legithostpital_router.router, prefix="/legit-hospital")
app.include_router(location_router.router, prefix="/location")
app.include_router(paper_router.router, prefix="/paper")
app.include_router(tag_router.router, prefix="/tag")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)