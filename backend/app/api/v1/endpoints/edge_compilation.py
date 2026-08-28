"""
ModelForge AI - Edge Compilation API Endpoints
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from app.services.edge_model_compiler_service import EdgeModelCompilerService

router = APIRouter()


class CompileModelRequest(BaseModel):
    model_version_id: str
    target_backend: str = Field("onnx_int8", description="onnx_int8, tensorrt_fp16, openvino, coreml")
    optimization_level: int = Field(3, ge=1, le=3)


@router.post("/compile")
async def compile_edge_model(request: CompileModelRequest):
    """Compile model artifact to optimized runtime graph."""
    try:
        res = EdgeModelCompilerService.compile_model(
            model_version_id=request.model_version_id,
            target_backend=request.target_backend,
            optimization_level=request.optimization_level,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Edge compilation failed: {str(e)}",
        )
