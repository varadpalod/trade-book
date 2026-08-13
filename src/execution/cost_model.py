class ExecutionCostModel:
    def __init__(self, sigma:float, eta:float, gamma:float, epsilon:float=0.0):
        if sigma<0 or eta<0 or gamma<0 or epsilon<0:
            raise ValueError("Values cant be negative ")

        self.sigma=sigma
        self.eta=eta
        self.gamma=gamma
        self.epsilon=epsilon

    def temporary_impact(self,quantity:float)->float:
        if quantity<0:
            raise ValueError("quantity cant be negative")

        return self.eta*quantity**2    

    def permenant_impact(self,quantity:float)->float:
        if quantity<0:
            raise ValueError("quantity cant be negative")
        return self.gamma*quantity

    def execution_cost(self,quantity:float)->float:
        if quantity<0:
            raise ValueError("quantity cant be negative")
        return self.temporary_impact(quantity)+self.permenant_impact(quantity)

    def expected_cost(self,trade_sizes:list[float])->float:
        if not trade_sizes:
            return 0.0

        cost=0.0
        for quantity in trade_sizes:
            if quantity<0:
                raise ValueError("quantity cant be negative")

            cost +=self.execution_cost(quantity)
        return cost 

    

    