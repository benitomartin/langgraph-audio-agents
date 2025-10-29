# import os
# import sys

# from loguru import logger as loguru_logger

# def setup_logging(log_level: str | None = None):
#     """Returns a logger configured for the current environment.

#     - Inside Prefect flow/task: Prefect's run logger (`logging.Logger`).
#     - Outside Prefect: Loguru logger.

#     Args:
#         log_level (str | None): Logging level to use (DEBUG, INFO, WARNING, ERROR).
#                                  Defaults to LOG_LEVEL env variable or DEBUG.

#     Returns:
#         logging.Logger | loguru.Logger: Configured logger instance.

#     """
#     log_level = log_level or os.getenv("LOG_LEVEL", "DEBUG").upper()


#     # Outside Prefect → Loguru
#     loguru_logger.remove()
#     loguru_logger.add(
#         sys.stdout,
#         level=log_level,
#         colorize=True,
#         backtrace=True,
#         diagnose=True,
#         format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
#         "<level>{level}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - "
#         "<level>{message}</level>",
#     )
#     loguru_logger.debug(f"Logging initialized at {log_level} level (Loguru).")
#     return loguru_logger
