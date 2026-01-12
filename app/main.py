"""主应用入口文件"""
from fastapi import FastAPI
from app.config import settings
from app.api import routes_ingest, routes_chat

app = FastAPI(
    title="Jarvis API",
    description="个人AI知识系统API",
    version="1.0.0"
)

# 注册路由
app.include_router(routes_ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(routes_chat.router, prefix="/api/v1", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "Jarvis API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )

