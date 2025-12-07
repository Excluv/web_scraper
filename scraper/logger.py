import sys
import logging
from pathlib import Path


class ScraperLogger:
    def __init__(self, 
                 module_name: str ="scraper_logger",
                 log_dir: str ="logs"):
        self.module_name = module_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logger()

    def _setup_logger(self):
        logging.config.dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)s",
                },
                "json": {
                    "format": "{'timestamp': '%(asctime)s', 'logger': '%(name)s', 'level': '%(levelname)s', \
                        'message': '%(message)s', 'file': '%(filename)s', 'line': '%(lineno)s', 'datefmt': '%Y-%m-%d %H:%M:%S'}"
                },
                "simple": {
                    "format": "%(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": sys.stdout, 
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": self.log_dir / f"{self.module_name}.log",
                    "maxBytes": 10 * 1024 * 1024, # 10MB
                    "backupCount": 5,
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": self.log_dir / f"{self.module_name}_errors.log",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                },
                "json_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": self.log_dir / f"{self.module_name}_json.log",
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 3,
                },
            },
            "loggers": {
                self.module_name: {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file", "json_file"],
                    "propagate": False,
                }
            }
        })
        self.logger = logging.getLogger(self.module_name)
    