import logging
import logging.handlers
import pytest
from src.logging_config import configure_logging, get_logger


class TestLoggingConfig:
    def teardown_method(self):
        """Her test sonrasi logger'i temizle (test isolation)."""
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

    def test_configure_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "logs"
        configure_logging(log_dir=str(log_dir), log_file="test.log", use_color=False)
        assert log_dir.exists()

    def test_log_message_written_to_file(self, tmp_path):
        configure_logging(log_dir=str(tmp_path), log_file="test.log", use_color=False)
        logger = get_logger("test_module")
        logger.info("Test mesaji 12345")

        for h in logging.getLogger().handlers:
            h.flush()

        log_file = tmp_path / "test.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Test mesaji 12345" in content
        assert "test_module" in content

    def test_debug_in_file_but_not_console(self, tmp_path):
        configure_logging(
            log_dir=str(tmp_path), log_file="test.log", level="INFO", use_color=False
        )
        logger = get_logger("test")
        logger.debug("DEBUG mesaji")
        logger.info("INFO mesaji")

        for h in logging.getLogger().handlers:
            h.flush()

        log_content = (tmp_path / "test.log").read_text(encoding="utf-8")
        assert "DEBUG mesaji" in log_content
        assert "INFO mesaji" in log_content

    def test_third_party_loggers_quieted(self, tmp_path):
        configure_logging(log_dir=str(tmp_path), log_file="test.log", use_color=False)
        assert logging.getLogger("yfinance").level   == logging.WARNING
        assert logging.getLogger("urllib3").level    == logging.WARNING
        assert logging.getLogger("matplotlib").level == logging.WARNING

    def test_idempotent_configuration(self, tmp_path):
        """Birden fazla cagri handler'lari ciftlemiyor mu?"""
        configure_logging(log_dir=str(tmp_path), log_file="test.log", use_color=False)
        first_count = len(logging.getLogger().handlers)

        configure_logging(log_dir=str(tmp_path), log_file="test.log", use_color=False)
        second_count = len(logging.getLogger().handlers)

        assert first_count == second_count

    def test_quiet_console_no_stderr_output(self, tmp_path, capsys):
        configure_logging(
            log_dir=str(tmp_path),
            log_file="test.log",
            quiet_console=True,
            use_color=False,
        )
        logger = get_logger("test")
        logger.info("Bu konsola cikmamali")
        logger.error("Bu da cikmamali")

        captured = capsys.readouterr()
        assert "Bu konsola cikmamali" not in captured.err
        assert "Bu da cikmamali" not in captured.err

        for h in logging.getLogger().handlers:
            h.flush()
        log_content = (tmp_path / "test.log").read_text(encoding="utf-8")
        assert "Bu konsola cikmamali" in log_content

    def test_rotation_config(self, tmp_path):
        """Rotasyon parametreleri dogru atanmis mi?"""
        configure_logging(
            log_dir=str(tmp_path),
            log_file="test.log",
            max_bytes=1024,
            backup_count=3,
            use_color=False,
        )
        rotating = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rotating) == 1
        assert rotating[0].maxBytes == 1024
        assert rotating[0].backupCount == 3
