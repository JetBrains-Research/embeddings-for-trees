from typing import Dict

from .logger import AbstractLogger, PrintLogger
from .file_logger import FileLogger
from .wandb_logger import WandBLogger

known_loggers: Dict[str, AbstractLogger.__class__] = {
    PrintLogger.name: PrintLogger,
    WandBLogger.name: WandBLogger,
    FileLogger.name: FileLogger
}


def create_logger(logger_name: str, log_dir: str, checkpoints_dir: str, config: Dict) -> AbstractLogger:
    if logger_name not in known_loggers.keys():
        raise ValueError(f"Unknown logger: {logger_name}, use one of {known_loggers.keys()}")
    logger_class = known_loggers[logger_name]
    return logger_class(log_dir, checkpoints_dir, config)
