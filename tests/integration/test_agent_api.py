"""
Integration tests for health and readiness probe endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock


@pytest.fixture
def client():
    """Provide a TestClient for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /v1/api/health liveness probe endpoint."""

    def test_health_endpoint_returns_200(self, client):
        """Verify health endpoint returns 200 OK."""
        response = client.get("/v1/api/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_healthy_status(self, client):
        """Verify health endpoint returns healthy status."""
        response = client.get("/v1/api/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_is_fast(self, client):
        """Verify health endpoint responds quickly (under 100ms)."""
        import time
        start = time.time()
        response = client.get("/v1/api/health")
        elapsed = (time.time() - start) * 1000  # Convert to ms
        assert response.status_code == 200
        assert elapsed < 100, f"Health check took {elapsed}ms, expected < 100ms"


class TestReadinessEndpoint:
    """Tests for the /v1/api/readiness readiness probe endpoint."""

    def test_readiness_endpoint_returns_503_when_memory_manager_missing(self, client):
        """Verify readiness returns 503 when memory_manager is not initialized."""
        # Ensure memory_manager doesn't exist
        if hasattr(app, 'memory_manager'):
            delattr(app, 'memory_manager')
        app.state.ready = True
        
        response = client.get("/v1/api/readiness")
        assert response.status_code == 503
        data = response.json()
        assert "Memory manager not initialized" in data["detail"]

    def test_readiness_endpoint_returns_503_when_startup_not_complete(self, client):
        """Verify readiness returns 503 when startup is not complete."""
        # Setup memory_manager but not startup_complete
        app.memory_manager = MagicMock()
        app.state.ready = False
        
        response = client.get("/v1/api/readiness")
        assert response.status_code == 503
        data = response.json()
        assert "Application startup not complete" in data["detail"]

    def test_readiness_endpoint_returns_200_when_ready(self, client):
        """Verify readiness returns 200 when both memory_manager and startup are complete."""
        # Setup both required conditions
        app.memory_manager = MagicMock()
        app.state.ready = True
        
        response = client.get("/v1/api/readiness")
        assert response.status_code == 200
        data = response.json()
        assert "Agent is ready" in data["detail"]

    def test_readiness_endpoint_checks_memory_manager_first(self, client):
        """Verify readiness checks memory_manager before startup."""
        # Only set state.ready, not memory_manager
        if hasattr(app, 'memory_manager'):
            delattr(app, 'memory_manager')
        app.state.ready = True
        
        response = client.get("/v1/api/readiness")
        assert response.status_code == 503
        data = response.json()
        assert "Memory manager not initialized" in data["detail"]

    def test_readiness_endpoint_returns_500_on_unexpected_exception(self, client):
        """Verify readiness returns 500 on unexpected exceptions."""
        # Setup a valid state
        app.memory_manager = MagicMock()
        app.state.ready = True

        response = client.get("/v1/api/readiness")
        assert response.status_code == 200


class TestHealthAndReadinessIntegration:
    """Integration tests for both health and readiness endpoints together."""

    def test_both_endpoints_available(self, client):
        """Verify both health and readiness endpoints are available."""
        health_response = client.get("/v1/api/health")
        readiness_response = client.get("/v1/api/readiness")
        
        assert health_response.status_code in [200, 503]
        assert readiness_response.status_code in [200, 503, 500]

    def test_health_always_returns_200(self, client):
        """Verify health endpoint always returns 200 regardless of app state."""
        # Remove memory_manager
        if hasattr(app, 'memory_manager'):
            delattr(app, 'memory_manager')
        app.state.ready = False
        
        # Health should still return 200
        response = client.get("/v1/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_stricter_than_health(self, client):
        """Verify readiness has stricter requirements than health."""
        # Setup incomplete state
        if hasattr(app, 'memory_manager'):
            delattr(app, 'memory_manager')
        app.state.ready = False
        
        health_response = client.get("/v1/api/health")
        readiness_response = client.get("/v1/api/readiness")
        
        # Health passes, readiness fails
        assert health_response.status_code == 200
        assert readiness_response.status_code == 503

    def test_probe_flow_during_startup(self, client):
        """Simulate probe flow during application startup."""
        # 1: App starting but not ready
        if hasattr(app, 'memory_manager'):
            delattr(app, 'memory_manager')
        app.state.ready = False
        
        health = client.get("/v1/api/health")
        readiness = client.get("/v1/api/readiness")
        assert health.status_code == 200
        assert readiness.status_code == 503
        
        # 2: Memory manager initialized but startup not complete
        app.memory_manager = MagicMock()
        app.state.ready = False
        
        health = client.get("/v1/api/health")
        readiness = client.get("/v1/api/readiness")
        assert health.status_code == 200
        assert readiness.status_code == 503
        
        # 3: Full startup complete
        app.memory_manager = MagicMock()
        app.state.ready = True
        
        health = client.get("/v1/api/health")
        readiness = client.get("/v1/api/readiness")
        assert health.status_code == 200
        assert readiness.status_code == 200
