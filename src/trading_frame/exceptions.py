"""Custom exceptions for Trading Frame."""


class InsufficientDataError(Exception):
    """
    Raised when there is not enough historical data to fill the frame.

    This typically occurs during prefill when require_full=True and
    the target timestamp is reached before accumulating enough periods.
    """
    pass
