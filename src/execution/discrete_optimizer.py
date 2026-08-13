import numpy as np
from scipy.optimize import minimize

class DiscreteOptimizer:

    def optimize(
        self,
        total_shares: int,
        intervals: int,
        execution_horizon: float,
        sigma: float,
        eta: float,
        gamma: float,
        lambda_risk: float,
    ) -> list[int]:

        if execution_horizon <= 0:
            raise ValueError("Execution horizon must be positive.")

        if total_shares < 0:
            raise ValueError("Total shares cannot be negative.")

        if intervals <= 0:
            raise ValueError("Intervals must be greater than zero.")

        if sigma < 0:
            raise ValueError("Sigma cannot be negative.")

        if eta < 0:
            raise ValueError("Eta cannot be negative.")

        if gamma < 0:
            raise ValueError("Gamma cannot be negative.")

        if lambda_risk < 0:
            raise ValueError("Lambda risk cannot be negative.")

        if total_shares == 0:
            return [0]*intervals

        tau = execution_horizon / intervals

        initial = np.array(
            self._initial_schedule(total_shares, intervals),
            dtype=float,
        )

        def objective(schedule):
            return self._objective(
                schedule=schedule.tolist(),
                total_shares=total_shares,
                sigma=sigma,
                eta=eta,
                gamma=gamma,
                lambda_risk=lambda_risk,
                tau=tau,
            )

        constraints = (
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - total_shares,
            },
        )

        bounds = [(0, None)] * intervals

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol":1e-9,"maxiter":200,"disp":False}
        )

        if not result.success:
            raise RuntimeError(result.message)

        schedule = np.round(result.x).astype(int)

        difference = total_shares - np.sum(schedule)
        schedule[-1] += difference
        
        return schedule.tolist()

    def _initial_schedule(self,total_shares:int,intervals:int,)->list[int]:
        base=total_shares//intervals
        remainder=total_shares%intervals
        schedule=[base]*intervals
        for i in range(remainder):
            schedule[i]+=1
        return schedule 
    

    def _compute_inventory(self,schedule:list[int],total_shares:int,)->list[int]:
        inventory=[total_shares]
        remaining=total_shares
        for trade in schedule:
            remaining -=trade
            inventory.append(remaining)
        
        return inventory 

    def _expected_cost(self,schedule:list[float],eta:float,gamma:float,tau:float,)->float:
        eta_tilde=eta-(gamma*tau/2)

        if eta_tilde<=0:
            raise ValueError("eta_tilde should be positive")

        return (eta_tilde/tau)*sum(trade**2 for trade in schedule)
    
    def _variance_cost(self,inventory:list[float],sigma:float,tau:float,)->float:
        return sigma**2*tau*sum(q**2 for q in inventory[:-1])

    def _objective(self,schedule:list[float],total_shares:int,sigma:float,eta:float,gamma:float,lambda_risk:float,tau:float,)->float:
        schedule=list(schedule)
        inventory=self._compute_inventory(schedule,total_shares,)

        expected = self._expected_cost(schedule, eta, gamma, tau)
        variance = self._variance_cost(inventory, sigma, tau)
        return expected+lambda_risk*variance