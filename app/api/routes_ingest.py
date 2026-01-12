"""文档摄入路由"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os

router = APIRouter()

@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """上传并处理单个文件"""
    # TODO: 实现文件处理逻辑
    return {"message": "File ingested", "filename": file.filename}

@router.post("/ingest/folder")
async def ingest_folder(folder_path: str):
    """处理文件夹中的所有文件"""
    # TODO: 实现文件夹处理逻辑
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"message": "Folder ingested", "folder_path": folder_path}

@router.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str):
    """获取摄入任务状态"""
    # TODO: 实现状态查询逻辑
    return {"job_id": job_id, "status": "completed"}

