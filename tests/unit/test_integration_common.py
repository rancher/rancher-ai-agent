import requests
import pytest

from tests.integration import common


class MockProcess:
    def __init__(self, exitcode=None):
        self.exitcode = exitcode

    def start(self):
        pass


def test_start_mock_mcp_servers_fails_when_process_exits(monkeypatch):
    process = MockProcess(exitcode=1)
    monkeypatch.setattr(common.multiprocessing, "Process", lambda **kwargs: process)

    with pytest.raises(RuntimeError, match="port 8080 exited with code 1"):
        common.start_mock_mcp_servers({8080: ("mock", [])})


def test_start_mock_mcp_servers_times_out_when_server_is_unavailable(monkeypatch):
    process = MockProcess()
    monkeypatch.setattr(common.multiprocessing, "Process", lambda **kwargs: process)

    def connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(common.requests, "get", connection_error)

    with pytest.raises(TimeoutError, match="waiting for mock MCP server on port 8080"):
        common.start_mock_mcp_servers({8080: ("mock", [])}, startup_timeout=0.01)
