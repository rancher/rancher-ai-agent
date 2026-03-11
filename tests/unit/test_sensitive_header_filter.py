import logging
import pytest
from app.main import _SensitiveHeaderFilter


@pytest.fixture
def log_filter():
    return _SensitiveHeaderFilter()


def _make_record(msg, args=None):
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg=msg,
        args=args,
        exc_info=None,
    )
    return record


class TestSensitiveHeaderFilter:
    def test_redacts_authorization_bearer_header(self, log_filter):
        msg = (
            "Sending http request: <AWSPreparedRequest stream_output=True, "
            "headers={'Content-Type': b'application/json', "
            "'Authorization': b'Bearer eyJhbGciOiJIUzI1NiJ9.secret_payload'}>"
        )
        record = _make_record(msg)
        assert log_filter.filter(record) is True
        assert "eyJhbGciOiJIUzI1NiJ9" not in record.msg
        assert "secret_payload" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_authorization_basic_header(self, log_filter):
        msg = "headers={'Authorization': b'Basic dXNlcjpwYXNz'}"
        record = _make_record(msg)
        log_filter.filter(record)
        assert "dXNlcjpwYXNz" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_redacts_x_api_key_header(self, log_filter):
        msg = "headers={'X-Api-Key': b'sk-live-abc123xyz'}"
        record = _make_record(msg)
        log_filter.filter(record)
        assert "sk-live-abc123xyz" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_does_not_alter_safe_messages(self, log_filter):
        msg = "Sending http request to https://example.com with headers={'Content-Type': b'application/json'}"
        record = _make_record(msg)
        log_filter.filter(record)
        assert record.msg == msg

    def test_always_returns_true(self, log_filter):
        """Filter should never suppress records, only redact them."""
        record = _make_record("'Authorization': b'Bearer secret'")
        assert log_filter.filter(record) is True

    def test_handles_record_with_args(self, log_filter):
        """If the record has args, they should be folded into msg before redaction."""
        msg = "Request headers: %s"
        args = ("{'Authorization': b'Bearer secret_token_value'}",)
        record = _make_record(msg, args=args)
        log_filter.filter(record)
        assert "secret_token_value" not in record.msg
        assert "[REDACTED]" in record.msg
        assert record.args is None

    def test_plain_authorization_header_line(self, log_filter):
        msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        record = _make_record(msg)
        log_filter.filter(record)
        assert "eyJhbGciOiJIUzI1NiJ9" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_case_insensitive_match(self, log_filter):
        msg = "headers={'authorization': b'Bearer mysecret123'}"
        record = _make_record(msg)
        log_filter.filter(record)
        assert "mysecret123" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_non_string_msg_is_untouched(self, log_filter):
        record = _make_record(12345)
        assert log_filter.filter(record) is True
        assert record.msg == 12345

    def test_handler_filter_catches_child_logger(self, log_filter):
        """Verify that the filter works when attached to a *handler* (not the
        logger), which is necessary to intercept records propagated from child
        loggers like ``botocore.endpoint``."""
        handler = logging.StreamHandler()
        handler.addFilter(log_filter)

        # Simulate a record from a child logger like botocore.endpoint
        record = logging.LogRecord(
            name="botocore.endpoint",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg=(
                "Sending http request: <AWSPreparedRequest "
                "headers={'Authorization': b'Bearer supersecret123'}>"
            ),
            args=None,
            exc_info=None,
        )
        # Handler.filter() returns the record (truthy) when all filters pass
        assert handler.filter(record)
        assert "supersecret123" not in record.msg
        assert "[REDACTED]" in record.msg
