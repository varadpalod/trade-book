import numpy as np 
class PriceProcess:
    def __init__(self,random_seed:int|None=None):
        self._rng=np.random.default_rng(random_seed)

    
    def generate(self,initial_price:float,sigma:float,intervals:float,tau:float,)->list[float]:
        if initial_price <= 0:
            raise ValueError("Initial price must be positive.")

        if sigma < 0:
            raise ValueError("Sigma cannot be negative.")

        if intervals <= 0:
            raise ValueError("Intervals must be positive.")

        if tau <= 0:
            raise ValueError("Tau must be positive.")

        prices = [initial_price]
        current_price = initial_price

        for _ in range(intervals):
            shock = sigma * np.sqrt(tau) * self._rng.normal()
            current_price += shock
            prices.append(current_price)

        return prices
