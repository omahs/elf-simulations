"""Helper functions for converting time units"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

import elfpy.types as types
from elfpy.math import FixedPoint, FixedPointMath


@dataclass
class BlockTime:
    r"""Global time."""

    time: float = 0  # time in years
    block_number: float = 0
    step_size: float = 1 / 365  # defaults to 1 day

    @property
    def time_in_seconds(self) -> float:
        """1 year = 31,556,952 seconds"""
        return self.time * 31_556_952

    def tick(self, delta_years: float) -> None:
        """ticks the time by delta_time amount"""
        self.time += delta_years

    def step(self) -> None:
        """ticks the time by step_size"""
        self.time += self.step_size

    def set_time(self, time: float) -> None:
        """Sets the time"""
        self.time = time

    def set_step_size(self, step_size: float) -> None:
        """Sets the step_size for tick"""
        self.step_size = step_size


@types.freezable(frozen=True, no_new_attribs=True)
@dataclass
class StretchedTime:
    r"""Stores time in units of days, as well as normalized & stretched variants

    .. todo:: Improve this constructor so that StretchedTime can be constructed from years.
    """
    days: float
    time_stretch: float
    normalizing_constant: float

    @property
    def stretched_time(self) -> float:
        r"""Returns days / normalizing_constant / time_stretch"""
        return days_to_time_remaining(self.days, self.time_stretch, normalizing_constant=self.normalizing_constant)

    @property
    def normalized_time(self) -> float:
        r"""Format time as normalized days"""
        return self.days / self.normalizing_constant

    @property
    def years(self) -> float:
        r"""Format time as normalized days"""
        return self.days / 365


def get_years_remaining(market_time: float, mint_time: float, position_duration_years: float) -> float:
    r"""Get the time remaining in years on a token

    Parameters
    ----------
    market_time: float
        Time that has elapsed in the given market, in years
    mint_time: float
        Time at which the token in question was minted, relative to market_time,
        in yearss. Should be less than market_time.
    position_duration_years: float
        Total duration of the token's term, in years

    Returns
    -------
    float
        Time left until token maturity, in years
    """
    if mint_time > market_time:
        raise ValueError(f"{mint_time=} must be less than {market_time=}.")
    years_elapsed = market_time - mint_time
    # if we are closing after the position duration has completed, then just set time_remaining to zero
    time_remaining = np.maximum(position_duration_years - years_elapsed, 0)
    return time_remaining


def norm_days(days: float, normalizing_constant: float = 365) -> float:
    r"""Returns days normalized, with a default assumption of a year-long scale

    Parameters
    ----------
    days: float
        Amount of days to normalize
    normalizing_constant: float
        Amount of days to use as a normalization factor. Defaults to 365

    Returns
    -------
    float
        Amount of days provided, converted to fractions of a year
    """
    return days / normalizing_constant


def days_to_time_remaining(days_remaining: float, time_stretch: float = 1, normalizing_constant: float = 365) -> float:
    r"""Converts remaining pool length in days to normalized and stretched time

    Parameters
    ----------
    days_remaining: float
        Time left until term maturity, in days
    time_stretch: float
        Amount of time units (in terms of a normalizing constant) to use for stretching time, for calculations
        Defaults to 1
    normalizing_constant: float
        Amount of days to use as a normalization factor
        Defaults to 365

    Returns
    -------
    float
        Time remaining until term maturity, in normalized and stretched time
    """
    normed_days_remaining = norm_days(days_remaining, normalizing_constant)
    return normed_days_remaining / time_stretch


def time_to_days_remaining(time_remaining: float, time_stretch: float = 1, normalizing_constant: float = 365) -> float:
    r"""Converts normalized and stretched time remaining in pool to days

    Parameters
    ----------
    time_remaining: float
        Time left until term maturity, in normalized and stretched time
    time_stretch: float
        Amount of time units (in terms of a normalizing constant) to use for stretching time, for calculations
        Defaults to 1
    normalizing_constant: float
        Amount of days to use as a normalization factor. Defaults to 365

    Returns
    -------
    float
        Time remaining until term maturity, in days
    """
    normed_days_remaining = time_remaining * time_stretch
    return normed_days_remaining * normalizing_constant


class TimeUnit(Enum):
    r"""Time units, with their parts in a year

    FP values are stored as parts in a year as FixedPoint 18 decimal integers"""
    SECONDS = FixedPoint(31_556_952 * 10**18)
    MINUTES = FixedPoint(525_600 * 10**18)
    HOURS = FixedPoint(8_760 * 10**18)
    DAYS = FixedPoint(365 * 10**18)
    YEARS = FixedPoint(1 * 10**18)


@dataclass
class BlockTimeFP:
    r"""State class for tracking block timestamps and global time

    .. todo::
    When we do the time refactor we need to think about whether it is better to use the Python time stdlib.
    This would make it easier to e.g. add/subtract times, but would require us to write converters that turn stdlib time
    into a FixedPoint representation from 0 (the start of simulation).
    Do the stdlib features make it worth the extra overhead?
    """

    _time: FixedPoint = FixedPoint(0)
    _block_number: FixedPoint = FixedPoint(0)
    _step_size: FixedPoint = FixedPoint("1.0") / FixedPoint("365.0")  # defaults to 1 day
    unit: TimeUnit = TimeUnit.YEARS

    def __post_init__(self):
        """.. todo::
        This is temporary until we do the time refactor.
        """
        if self.unit != TimeUnit.YEARS:
            raise ValueError(f"Only `years` is supported for {self.unit.name}")

    def tick(self, delta_years: FixedPoint) -> None:
        """ticks the time by delta_time amount"""
        self._time += delta_years

    def step(self) -> None:
        """ticks the time by step_size"""
        self._time += self.step_size

    def time_conversion(self, unit: TimeUnit = TimeUnit.SECONDS) -> FixedPoint:
        """Convert time to different units

        .. todo:: For now this only converts from years to `unit`.
        We will need to add conditions for self.unit in each conversion type
        """
        return self.time * unit.value  # FixedPoint(unit)

    @property
    def time(self):
        """Get the time"""
        return self._time

    @time.setter
    def time(self, value):
        """Private time attribute and time property cannot be set directly,
        since we want to check types using a setter."""
        raise AttributeError("time is a read-only attribute; use `set_time()` to adjust")

    def set_time(self, time: FixedPoint, unit: TimeUnit) -> None:
        """Sets the time"""
        if not isinstance(time, FixedPoint):
            raise TypeError(f"{time=} must be a FixedPoint variable")
        self._time = time
        if unit != TimeUnit.YEARS:
            raise NotImplementedError(f"unit must be TimeUnit.YEARS, not {unit=}.")
        self.unit = unit

    @property
    def block_number(self):
        """Get the block_number"""
        return self._block_number

    @block_number.setter
    def block_number(self, value):
        """Private block_number attribute and block_number property cannot be set directly,
        since we want to check types using a setter."""
        raise AttributeError("block_number is a read-only attribute; use `set_block_number()` to adjust")

    def set_block_number(self, block_number: FixedPoint) -> None:
        """Sets the block_number"""
        if not isinstance(block_number, FixedPoint):
            raise TypeError(f"{block_number=} must be a FixedPoint variable")
        self._block_number = block_number

    @property
    def step_size(self):
        """Get the step_size"""
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        """Private block_number attribute and block_number property cannot be set directly,
        since we want to check types using a setter."""
        raise AttributeError("step_size is a read-only attribute; use `set_step_size()` to adjust")

    def set_step_size(self, step_size: FixedPoint) -> None:
        """Sets the step_size for tick"""
        if not isinstance(step_size, FixedPoint):
            raise TypeError(f"{step_size=} must be a FixedPoint variable")
        self._step_size = step_size


@types.freezable(frozen=True, no_new_attribs=True)
@dataclass
class StretchedTimeFP:
    r"""Stores time in units of days, as well as normalized & stretched variants

    .. todo:: Improve this constructor so that StretchedTime can be constructed from years.
    """
    days: FixedPoint
    time_stretch: FixedPoint
    normalizing_constant: FixedPoint

    @property
    def stretched_time(self) -> FixedPoint:
        r"""Returns days / normalizing_constant / time_stretch"""
        return days_to_time_remaining_fp(self.days, self.time_stretch, normalizing_constant=self.normalizing_constant)

    @property
    def normalized_time(self) -> FixedPoint:
        r"""Format time as normalized days"""
        return self.days / self.normalizing_constant

    @property
    def years(self) -> FixedPoint:
        r"""Format time as normalized days"""
        return self.days / FixedPoint("365.0")


def get_years_remaining_fp(
    market_time: FixedPoint, mint_time: FixedPoint, position_duration_years: FixedPoint
) -> FixedPoint:
    r"""Get the time remaining in years on a token

    Parameters
    ----------
    market_time: FixedPoint
        Time that has elapsed in the given market, in years.
    mint_time: FixedPoint
        Time at which the token in question was minted, relative to market_time,
        in yearss. Should be less than market_time.
    position_duration_years: FixedPoint
        Total duration of the token's term, in years

    Returns
    -------
    FixedPoint
        Time left until token maturity, in years
    """
    if mint_time > market_time:
        raise ValueError(f"{mint_time=} must be less than {market_time=}.")
    years_elapsed = market_time - mint_time
    # if we are closing after the position duration has completed, then just set time_remaining to zero
    time_remaining = FixedPointMath.maximum(position_duration_years - years_elapsed, FixedPoint(0))
    return time_remaining


def days_to_time_remaining_fp(
    days_remaining: FixedPoint,
    time_stretch: FixedPoint = FixedPoint("1.0"),
    normalizing_constant: FixedPoint = FixedPoint("365.0"),
) -> FixedPoint:
    r"""Converts remaining pool length in days to normalized and stretched time

    Parameters
    ----------
    days_remaining: FixedPoint
        Time left until term maturity, in days.
    time_stretch: FixedPoint
        Amount of time units (in terms of a normalizing constant) to use for stretching time, for calculations
        Defaults to FixedPoint("1.0").
    normalizing_constant: FixedPoint
        Amount of days to use as a normalization factor.
        Defaults to FixedPoint("365.0")

    Returns
    -------
    FixedPoint
        Time remaining until term maturity, in normalized and stretched time
    """
    normed_days_remaining = days_remaining / normalizing_constant
    return normed_days_remaining / time_stretch
