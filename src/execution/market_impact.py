class MarketImpact:

    def __init__(self,eta:float,gamma:float):
        self.eta=eta
        self.gamma=gamma

    def permanent_impact(self,trade_size:float,)->float:
        if self.gamma<0:
            raise ValueError("gamma cant be negative")

        return self.gamma*trade_size

    def temporary_impact(self,trade_size:float,tau:float,)->float:
        if self.eta<0:
            raise ValueError("eta cant be negative")
        if(tau<=0):
            raise ValueError("tau cant be negative")

        return self.eta*trade_size/tau
