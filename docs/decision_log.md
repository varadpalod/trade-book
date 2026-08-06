Decision 001: When total shares are not perfectly divisible by the number of intervals, distribute the remainder across the earliest intervals to ensure all shares are scheduled.

Decision 002: Volume profiles are considered valid if their weights sum to 1.0 within a small floating-point tolerance.

This makes the algorithm independent of the absolute market volume.


Decision 003

VWAP accepts normalized volume weights instead of raw market volume.
This keeps the algorithm independent of absolute market activity and
simplifies comparisons across different datasets.