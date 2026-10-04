import numpy as np 
class PriceProcess:
    def __init__(self,random_seed:int|None=None):
        self._rng=np.random.default_rng(random_seed)

    
    def generate_path(self,initial_price:float,sigma:float,intervals:float,tau:float,)->list[float]:
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
        mu=0.0

        for _ in range(intervals):
            z=self._rng.normal()
            current_price = current_price*np.exp((mu-0.5*sigma**2)*tau+sigma*np.sqrt(tau)*z)
            prices.append(current_price)

        return prices
