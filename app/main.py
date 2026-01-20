"""主应用入口文件"""
from fastapi import FastAPI
from app.config import settings
from app.api import routes_ingest, routes_chat
from app.vector.qdrant import get_client
import redis
import psycopg2

app = FastAPI(
    title="Jarvis V1",
    description="个人AI知识系统",
    version="1.0.0"
)

# 注册路由
app.include_router(routes_ingest.router, prefix="/api/v1", tags=["ingest"])
app.include_router(routes_chat.router, prefix="/api/v1", tags=["chat"])


def qdrant_ok() -> bool:
    client = get_client()
    collections = client.get_collections()
    return True
def redis_ok() -> bool:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=1
    )
    try:
        redis_client.ping()
    except redis.exceptions.ConnectionError:
        return False
    return True
def postgres_ok() -> bool:
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        connect_timeout=2
    )
    conn.close()
    return True

@app.get("/")
async def root():
    return {"message": "Jarvis API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/infra/health")
async def infra_health():
    status = {
        "qdrant": False ,
        "redis": False,
        "postgres": False,
    }
    errors = {}
    for name, fn in [
        ("qdrant", qdrant_ok),
        ("redis", redis_ok),
        ("postgres", postgres_ok),
    ]:
        try:
            status[name] = fn()
        except Exception as e:
            errors[name] = str(e)
    return {
        "status":status,
        "errors":errors,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG
    )

