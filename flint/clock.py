"""Wall-clock session logic for US equity markets, with the exchange calendar.

The wall clock knows weekdays and hours; the calendar (`set_calendar`, from Finnhub's
market-holiday list, persisted by the engine in `state/holidays.json`) knows the full holidays
and the early closes, so a Friday after Thanksgiving ends at 13:00 and a Good Friday never
opens. The engine's session gates ask both, and the live Finnhub status as a third voice: a
trade needs the clock, the calendar and the status to agree that the regular session is on.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
OPEN_MIN, CLOSE_MIN = 570, 960                       # 09:30 and 16:00 ET
_CALENDAR: dict[str, tuple[int, int] | None] = {}   # ISO date -> (open, close) minutes of a shortened day, None for a closed one


def set_calendar(days: dict) -> None:
    """Install the exchange calendar: {ISO date: "HH:MM-HH:MM" for an early close, "" or None
    for a day the exchange is closed} (Finnhub's `tradingHour`)."""
    cal: dict[str, tuple[int, int] | None] = {}
    for day, hours in (days or {}).items():
        hrs = None
        if hours:
            try:
                a, b = str(hours).split("-")
                h1, m1 = a.split(":"); h2, m2 = b.split(":")
                hrs = (int(h1) * 60 + int(m1), int(h2) * 60 + int(m2))
            except ValueError:
                hrs = None
        cal[str(day)[:10]] = hrs
    _CALENDAR.clear()
    _CALENDAR.update(cal)


def calendar() -> dict:
    return dict(_CALENDAR)


def session_hours(now=None) -> tuple[int, int] | None:
    """The regular session on `now`'s date in minutes after midnight ET: (570, 960) on an
    ordinary weekday, the shortened hours on an early close, None on a weekend or a holiday."""
    if now is None:
        now = datetime.now(NY)
    now_ny = now.astimezone(NY)
    if now_ny.weekday() >= 5:
        return None
    key = now_ny.date().isoformat()
    if key in _CALENDAR:
        return _CALENDAR[key]
    return (OPEN_MIN, CLOSE_MIN)


def regular_session(now=None) -> bool:
    """Regular US equity session by the wall clock and the calendar: 9:30-16:00 ET on a
    weekday that is not an exchange holiday, the shortened hours on an early close.

    Args:
        now: An aware datetime in any timezone. If None, uses the current time
             in America/New_York.
    """
    if now is None:
        now = datetime.now(NY)
    now_ny = now.astimezone(NY)
    hrs = session_hours(now_ny)
    if hrs is None:
        return False
    mins = now_ny.hour * 60 + now_ny.minute
    return hrs[0] <= mins < hrs[1]


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
    now_ny = now.astimezone(NY)
    hrs = session_hours(now_ny)
    if hrs is None:
        return False                                  # a holiday has no extended session either
    mins = now_ny.hour * 60 + now_ny.minute
    return 240 <= mins < hrs[0] or hrs[1] <= mins < 1200


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
