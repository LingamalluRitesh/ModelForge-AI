"""
ModelForge AI SDK - Datasets Manager
"""

import os
from typing import Any, Dict, List, Optional
import io


class DatasetVersion:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def version_tag(self) -> str:
        return self._data.get("version_tag", "")

    @property
    def row_count(self) -> int:
        return self._data.get("row_count", 0)

    @property
    def column_count(self) -> int:
        return self._data.get("column_count", 0)


class Dataset:
    def __init__(self, data: Dict[str, Any], manager: "DatasetManager"):
        self._data = data
        self._manager = manager

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def format(self) -> str:
        return self._data.get("format", "")

    @property
    def target_column(self) -> Optional[str]:
        return self._data.get("target_column")

    def get_quality_report(self) -> Dict[str, Any]:
        """Fetch automated data quality gate score and rule checks."""
        ver_id = self._data.get("latest_version", {}).get("id")
        if not ver_id:
            raise ValueError("Dataset has no uploaded versions.")
        return self._manager.client.request("GET", f"/datasets/versions/{ver_id}/quality").get("data", {})

    def get_profile_report(self) -> Dict[str, Any]:
        """Fetch statistical column distributions, quantiles, and correlation matrix."""
        ver_id = self._data.get("latest_version", {}).get("id")
        if not ver_id:
            raise ValueError("Dataset has no uploaded versions.")
        return self._manager.client.request("GET", f"/datasets/versions/{ver_id}/profile").get("data", {})


class DatasetManager:
    def __init__(self, client):
        self.client = client

    def list(self, project_id: str) -> List[Dataset]:
        """List datasets within a project."""
        res = self.client.request("GET", f"/datasets?project_id={project_id}")
        return [Dataset(d, self) for d in res.get("data", [])]

    def upload(
        self,
        project_id: str,
        name: str,
        file_path_or_buffer: Any,
        target_column: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dataset:
        """Upload tabular CSV, Parquet, or JSON dataset with automated quality analysis."""
        if isinstance(file_path_or_buffer, str):
            filename = os.path.basename(file_path_or_buffer)
            with open(file_path_or_buffer, "rb") as f:
                content = f.read()
        elif hasattr(file_path_or_buffer, "read"):
            filename = getattr(file_path_or_buffer, "name", "dataset.csv")
            content = file_path_or_buffer.read()
        else:
            raise ValueError("Expected file path string or binary file buffer.")

        files = {"file": (filename, content)}
        data = {
            "project_id": project_id,
            "name": name,
            "target_column": target_column or "",
            "description": description or "",
        }
        res = self.client.request(
            "POST",
            "/datasets/upload",
            files=files,
            params=data,
        )
        return Dataset(res.get("data", {}), self)
