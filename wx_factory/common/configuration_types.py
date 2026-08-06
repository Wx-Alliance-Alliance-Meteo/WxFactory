from typing import TypeVar

OptionType = TypeVar("OptionType", bound=str | int | float | list[int] | list[float] | bool)
