"""
ModelForge AI - Unified Object Storage Engine
Supports Local File Storage, Amazon S3, and MinIO object storage.
Handles datasets, model checkpoints, pickled pipelines, and evaluation reports.
"""

import os
import shutil
from pathlib import Path
from typing import BinaryIO, Optional, Union
import aiofiles
from app.core.config import settings
from app.core.logging import logger


class StorageEngine:
    """Unified storage interface abstracting local disk and S3/MinIO cloud storage."""

    def __init__(self):
        self.provider = settings.STORAGE_PROVIDER.lower()
        self.local_root = Path(settings.STORAGE_LOCAL_PATH)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._s3_client = None

        if self.provider in ("s3", "minio"):
            self._init_s3_client()

    def _init_s3_client(self):
        """Lazy-initialize boto3 S3 client."""
        try:
            import boto3
            from botocore.client import Config

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL if self.provider == "minio" else None,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
                region_name=settings.S3_REGION,
                config=Config(signature_version="s3v4"),
            )
            # Ensure bucket exists
            try:
                self._s3_client.head_bucket(Bucket=settings.S3_BUCKET_NAME)
            except Exception:
                self._s3_client.create_bucket(Bucket=settings.S3_BUCKET_NAME)
        except Exception as e:
            logger.warning(f"Failed to initialize S3/MinIO client: {e}. Falling back to local storage.")
            self.provider = "local"

    async def save_file(self, file_content: bytes, destination_key: str) -> str:
        """
        Save binary content to storage.
        Returns the resolved storage URI (e.g. 'local:///path/to/file' or 's3://bucket/key').
        """
        # Sanitize key
        destination_key = destination_key.lstrip("/\\")

        if self.provider in ("s3", "minio") and self._s3_client:
            try:
                self._s3_client.put_object(
                    Bucket=settings.S3_BUCKET_NAME,
                    Key=destination_key,
                    Body=file_content,
                )
                return f"s3://{settings.S3_BUCKET_NAME}/{destination_key}"
            except Exception as e:
                logger.error(f"S3 upload failed: {e}. Falling back to local disk.")

        # Local storage fallback / default
        local_path = self.local_root / destination_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(local_path, "wb") as f:
            await f.write(file_content)
        return str(local_path.resolve())

    async def read_file(self, storage_uri: str) -> bytes:
        """Read binary content from a storage URI."""
        if storage_uri.startswith("s3://") and self._s3_client:
            parts = storage_uri.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1]
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        else:
            path = Path(storage_uri)
            if not path.is_absolute():
                path = self.local_root / storage_uri
            async with aiofiles.open(path, "rb") as f:
                return await f.read()

    def get_local_path(self, storage_uri: str) -> str:
        """Return the local filesystem path if available."""
        if storage_uri.startswith("s3://"):
            # Download to local cache directory if needed
            local_cache = self.local_root / "s3_cache" / storage_uri.replace("s3://", "")
            local_cache.parent.mkdir(parents=True, exist_ok=True)
            if not local_cache.exists():
                parts = storage_uri.replace("s3://", "").split("/", 1)
                self._s3_client.download_file(parts[0], parts[1], str(local_cache))
            return str(local_cache.resolve())
        return str(Path(storage_uri).resolve())

    async def delete_file(self, storage_uri: str) -> bool:
        """Delete an object from storage."""
        try:
            if storage_uri.startswith("s3://") and self._s3_client:
                parts = storage_uri.replace("s3://", "").split("/", 1)
                self._s3_client.delete_object(Bucket=parts[0], Key=parts[1])
                return True
            else:
                path = Path(storage_uri)
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                return True
        except Exception as e:
            logger.error(f"Failed to delete storage file '{storage_uri}': {e}")
            return False


storage_engine = StorageEngine()
