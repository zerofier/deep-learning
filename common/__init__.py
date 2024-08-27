from abc import ABC, abstractmethod
from typing import Any, Tuple

from tensorflow import Tensor


class Agent(ABC):
    @abstractmethod
    def action_sample(self, observation) -> Tuple[Any]:
        pass
