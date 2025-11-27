"""
Airflow REST API client.
Wraps Airflow's REST API for DAG management.
"""

from datetime import datetime
from typing import Any

import httpx

from app.core.config import settings
from app.core.errors import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AirflowClient:
    """Client for Airflow REST API."""

    def __init__(self):
        """Initialize the Airflow client."""
        self.base_url = settings.airflow_base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/v1"
        self.auth = (settings.airflow_username, settings.airflow_password)

    def _get_client(self) -> httpx.Client:
        """Get an HTTP client."""
        return httpx.Client(
            base_url=self.api_url,
            auth=self.auth,
            timeout=30.0,
        )

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get an async HTTP client."""
        return httpx.AsyncClient(
            base_url=self.api_url,
            auth=self.auth,
            timeout=30.0,
        )

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response."""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Airflow API error: {e.response.status_code} - {e.response.text}")
            raise ExternalServiceError("Airflow", str(e))
        except Exception as e:
            logger.error(f"Airflow client error: {e}")
            raise ExternalServiceError("Airflow", str(e))

    # =========================================================================
    # DAG Operations
    # =========================================================================

    def list_dags(self) -> list[dict[str, Any]]:
        """
        List all DAGs.
        
        Returns:
            List of DAG info dicts
        """
        with self._get_client() as client:
            response = client.get("/dags")
            data = self._handle_response(response)
            return data.get("dags", [])

    def get_dag(self, dag_id: str) -> dict[str, Any]:
        """
        Get DAG details.
        
        Args:
            dag_id: DAG identifier
            
        Returns:
            DAG info dict
        """
        with self._get_client() as client:
            response = client.get(f"/dags/{dag_id}")
            return self._handle_response(response)

    def pause_dag(self, dag_id: str, paused: bool = True) -> dict[str, Any]:
        """
        Pause or unpause a DAG.
        
        Args:
            dag_id: DAG identifier
            paused: True to pause, False to unpause
            
        Returns:
            Updated DAG info
        """
        with self._get_client() as client:
            response = client.patch(
                f"/dags/{dag_id}",
                json={"is_paused": paused},
            )
            return self._handle_response(response)

    def trigger_dag_run(
        self,
        dag_id: str,
        run_id: str | None = None,
        conf: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Trigger a new DAG run.
        
        Args:
            dag_id: DAG identifier
            run_id: Optional run ID
            conf: Optional configuration dict
            
        Returns:
            DAG run info
        """
        payload: dict[str, Any] = {}
        if run_id:
            payload["dag_run_id"] = run_id
        if conf:
            payload["conf"] = conf

        with self._get_client() as client:
            response = client.post(f"/dags/{dag_id}/dagRuns", json=payload)
            return self._handle_response(response)

    # =========================================================================
    # DAG Run Operations
    # =========================================================================

    def list_dag_runs(
        self,
        dag_id: str,
        limit: int = 25,
        offset: int = 0,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List DAG runs.
        
        Args:
            dag_id: DAG identifier
            limit: Max results
            offset: Results offset
            state: Filter by state
            
        Returns:
            List of DAG run info dicts
        """
        params = {"limit": limit, "offset": offset}
        if state:
            params["state"] = state

        with self._get_client() as client:
            response = client.get(f"/dags/{dag_id}/dagRuns", params=params)
            data = self._handle_response(response)
            return data.get("dag_runs", [])

    def get_dag_run(self, dag_id: str, run_id: str) -> dict[str, Any]:
        """
        Get DAG run details.
        
        Args:
            dag_id: DAG identifier
            run_id: Run identifier
            
        Returns:
            DAG run info
        """
        with self._get_client() as client:
            response = client.get(f"/dags/{dag_id}/dagRuns/{run_id}")
            return self._handle_response(response)

    def stop_dag_run(self, dag_id: str, run_id: str) -> dict[str, Any]:
        """
        Mark a DAG run as failed (stop it).
        
        Args:
            dag_id: DAG identifier
            run_id: Run identifier
            
        Returns:
            Updated DAG run info
        """
        with self._get_client() as client:
            response = client.patch(
                f"/dags/{dag_id}/dagRuns/{run_id}",
                json={"state": "failed"},
            )
            return self._handle_response(response)

    # =========================================================================
    # Task Operations
    # =========================================================================

    def list_task_instances(
        self,
        dag_id: str,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """
        List task instances for a DAG run.
        
        Args:
            dag_id: DAG identifier
            run_id: Run identifier
            
        Returns:
            List of task instance info dicts
        """
        with self._get_client() as client:
            response = client.get(
                f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
            )
            data = self._handle_response(response)
            return data.get("task_instances", [])

    def get_task_logs(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        try_number: int = 1,
    ) -> str:
        """
        Get task logs.
        
        Args:
            dag_id: DAG identifier
            run_id: Run identifier
            task_id: Task identifier
            try_number: Attempt number
            
        Returns:
            Log content as string
        """
        with self._get_client() as client:
            response = client.get(
                f"/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}",
                headers={"Accept": "text/plain"},
            )
            response.raise_for_status()
            return response.text


# Singleton instance
airflow_client = AirflowClient()

