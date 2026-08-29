import pickle
import pytest
from backend.app.services.artifact_integrity_service import ArtifactIntegrityService


def test_safe_artifact_signing_and_verification():
    service = ArtifactIntegrityService()
    safe_data = pickle.dumps({"weights": [0.1, 0.2, 0.3], "architecture": "linear"})

    fingerprint = service.compute_artifact_fingerprint(safe_data)
    assert len(fingerprint["sha384_checksum"]) == 96
    assert len(fingerprint["hmac_signature"]) == 96

    assert service.verify_artifact_signature(safe_data, fingerprint["hmac_signature"]) is True
    assert service.verify_artifact_signature(safe_data, "fake_signature") is False

    scan = service.scan_pickle_bytecode(safe_data)
    assert scan["is_safe_for_loading"] is True
    assert len(scan["detected_threats"]) == 0
