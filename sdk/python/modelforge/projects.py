"""
ModelForge AI SDK - Projects Manager
"""

from typing import Any, Dict, List, Optional


class Project:
    def __init__(self, data: Dict[str, Any], manager: "ProjectManager"):
        self._data = data
        self._manager = manager

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def slug(self) -> str:
        return self._data.get("slug", "")

    @property
    def problem_type(self) -> str:
        return self._data.get("problem_type", "")

    @property
    def description(self) -> Optional[str]:
        return self._data.get("description")

    def __repr__(self) -> str:
        return f"<ModelForge Project id='{self.id}' name='{self.name}' type='{self.problem_type}'>"


class ProjectManager:
    def __init__(self, client):
        self.client = client

    def list(self) -> List[Project]:
        """List all projects available in active organization."""
        res = self.client.request("GET", "/projects")
        return [Project(p, self) for p in res.get("data", [])]

    def get(self, project_id: str) -> Project:
        """Get project details by ID or Slug."""
        res = self.client.request("GET", f"/projects/{project_id}")
        return Project(res.get("data", {}), self)

    def create(
        self,
        name: str,
        problem_type: str = "classification",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Project:
        """Create a new enterprise ML project."""
        payload = {
            "name": name,
            "problem_type": problem_type,
            "description": description,
            "tags": tags or [],
        }
        res = self.client.request("POST", "/projects", json_data=payload)
        return Project(res.get("data", {}), self)
