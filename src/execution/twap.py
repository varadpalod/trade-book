from .strategy import ExecutionStrategy

class TWAPStrategy(ExecutionStrategy):
    def generate_schedule(self, total_shares:int, intervals:int, market_data=None,)->list[int]:
        if total_shares<0:
            raise ValueError("total shares cant be negative")

        if intervals <=0:
            raise ValueError("intervals must be greater than zero")

        if total_shares==0:
            return []

        shares_per_interval=total_shares//intervals
        remainder=total_shares%intervals

        schedule=[shares_per_interval]*intervals

        for i in range(remainder):
            schedule[i] += 1

        return schedule
