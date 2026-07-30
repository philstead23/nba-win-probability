"""Small shared helpers: logging setup and NBA clock string parsing."""

import logging
import re

CLOCK_PATTERN = re.compile(r"PT(\d+)M([\d.]+)S")


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def parse_clock(clock_str: str) -> float:
    """Convert an ISO-8601 duration like 'PT11M23.00S' into seconds remaining (683.0)."""
    match = CLOCK_PATTERN.match(clock_str)
    if not match:
        raise ValueError(f"Unrecognized clock format: {clock_str!r}")
    minutes, seconds = match.groups()
    return int(minutes) * 60 + float(seconds)
