from .strategy import ExecutionStrategy 

class VWAPStrategy(ExecutionStrategy):
    def generate_schedule(self, total_shares, intervals, market_data=None)->list[int]:
        if total_shares<0:
            raise ValueError("shares cant be negative")

        if intervals <=0:
            raise ValueError("intervals must be greater than zero")
        
        if total_shares==0:
            return []

        if market_data is None:
            raise ValueError("VWAP needs market data")

        if len(market_data) != intervals:
            raise ValueError("Volume profile length must equal the number of intervals." )

        if any(weight < 0 for weight in market_data):
            raise ValueError("Volume weights cannot be negative.")

        if abs(sum(market_data) - 1.0) > 1e-9:
            raise ValueError("volume profile must sum to 1")

        schedule = []

        for weight in market_data:
            shares = int(total_shares * weight)
            schedule.append(shares)

        allocated=sum(schedule)
        remainder=total_shares-allocated

        for i in range(remainder):
            schedule[i] += 1

        return schedule

