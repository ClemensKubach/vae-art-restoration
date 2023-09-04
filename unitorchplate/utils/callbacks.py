from __future__ import annotations
from typing import TYPE_CHECKING

from lightning import Callback
from strenum import StrEnum

if TYPE_CHECKING:
    from unitorchplate.runner.run import Run


class Callbacks(StrEnum):

    def instance(self, run: Run) -> Callback | None:
        return None
