#abc stands for abstarct base classes 
from abc import ABC, abstractmethod

class ExecutionStrategy(ABC):
    @abstractmethod
    def generate_schedule(self,total_shares: int,intervals: int,market_data=None,) -> list[int]:
        pass

