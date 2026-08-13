class MarketImpact:
    def permanent_impact(self,trade_size:float,gamma:float,)->float:
        if gamma<0:
            raise ValueError("gamma cant be negative")

        return gamma*trade_size

    def temporary_impact(self,trade_size:float,eta:float,)->float:
        if eta<0:
            raise ValueError("eta cant be negative")

        return eta*trade_size
