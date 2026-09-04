"""Wall-clock session logic for US equity markets."""
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")


def regular_session(now=None) -> bool:
    """Regular US equity session by the wall clock (9:30-16:00 ET, weekdays).

    Gates trading; a no-trade (volume 0) bar additionally covers holidays and
    early closes.

    Args:
        now: An aware datetime in any timezone. If None, uses the current time
             in America/New_York.

    Returns:
        True if the time falls within 9:30-16:00 ET on a weekday, False otherwise.
    """
    if now is None:
        now = datetime.now(NY)
    # Convert to NY timezone for consistent hour/minute extraction
    now_ny = now.astimezone(NY)
    if now_ny.weekday() >= 5:
        return False
    mins = now_ny.hour * 60 + now_ny.minute
    return 570 <= mins < 960


def extended_session(now=None) -> bool:
    """Schwab's extended sessions by the wall clock: 4:00-9:30 and 16:00-20:00 ET, weekdays.

    Stock only -- options do not trade here, so puts and straddles keep the
    regular gate.

    Args:
        now: An aware datetime in any timezone. If None, uses the current time
             in America/New_York.

    Returns:
        True if the time falls within 4:00-9:30 or 16:00-20:00 ET on a weekday,
        False otherwise.
    """
    if now is None:
        now = datetime.now(NY)
    # Convert to NY timezone for consistent hour/minute extraction
    now_ny = now.astimezone(NY)
    if now_ny.weekday() >= 5:
        return False
    mins = now_ny.hour * 60 + now_ny.minute
    return 240 <= mins < 570 or 960 <= mins < 1200


def stock_session(extended_hours: bool, now=None) -> bool:
    """When a long-stock entry or exit is allowed.

    This is the regular session (9:30-16:00 ET), plus the extended sessions
    (4:00-9:30 and 16:00-20:00 ET) when `extended_hours` is True.

    Args:
        extended_hours: If True, include extended pre-market and after-hours.
        now: An aware datetime in any timezone. If None, uses the current time
             in America/New_York.

    Returns:
        True if stock trading is allowed at the given time, False otherwise.
    """
    if now is None:
        now = datetime.now(NY)
    return regular_session(now) or (extended_hours and extended_session(now))
