"""
Model Artifact Integrity REST Endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from app.services.artifact_integrity_service import ArtifactIntegrityService

router = APIRouter(prefix="/artifact-security", tags=["Artifact Integrity Security"])
security_service = ArtifactIntegrityService()


class VerifyRequest(BaseModel):
    payload_hex: str
    signature: str


@router.post("/scan-and-sign", summary="Scan and Sign Model Artifact")
def scan_and_sign(payload_hex: str) -> Dict[str, Any]:
    try:
        raw_bytes = bytes.fromhex(payload_hex)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hexadecimal payload representation.")

    scan = security_service.scan_pickle_bytecode(raw_bytes)
    fingerprint = security_service.compute_artifact_fingerprint(raw_bytes)

    return {
        "security_scan": scan,
        "fingerprint": fingerprint,
    }


@router.post("/verify", summary="Verify Model Artifact HMAC Signature")
def verify_signature(req: VerifyRequest) -> Dict[str, Any]:
    try:
        raw_bytes = bytes.fromhex(req.payload_hex)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hexadecimal payload representation.")

    valid = security_service.verify_artifact_signature(raw_bytes, req.signature)
    return {"valid": valid}
