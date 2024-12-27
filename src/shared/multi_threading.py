import os

from shared.logger import logger


def get_available_threads():
    """Get number of available CPU threads that can be used for parallel processing."""

    try:
        num_threads = os.cpu_count()  # This is not 100% the best approach but os.sched_getaffinity does not work on Windows.

    except NotImplementedError:
        logger.warning("Automatic thread detection didn't work. Defaulting to 1 thread only.")
        num_threads = 1

    logger.info(f"\n[Status] Using {num_threads} CPU threads.")
    return num_threads
