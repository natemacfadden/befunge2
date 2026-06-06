"""
Interface from model to benchmark.
"""

from typing import Protocol


class Interface(Protocol):
    model: str      # what model it wraps
    language: str   # what language the output program is written in

    def propose(self, seq: list[int], k: int) -> list[str]:
        """Given a sequence of outputs, propose k functions that print them.
        0th element of the list is the 'top guess'."""
        ...
