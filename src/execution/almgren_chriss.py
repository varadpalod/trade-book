from .strategy import ExecutionStrategy


class AlmgrenChrissStrategy(ExecutionStrategy):
    ##simplified hai proper wala likhna padega after market simulation 
    def __init__(self, lambda_risk: float = 0.1):
        if lambda_risk < 0:
            raise ValueError("lambda_risk cannot be negative.")

        self.lambda_risk = lambda_risk

    def generate_schedule(self,total_shares: int,intervals: int,market_data=None,) -> list[int]:

        if total_shares < 0:
            raise ValueError("Total shares cannot be negative.")

        if intervals <= 0:
            raise ValueError("Intervals must be greater than zero.")

        if total_shares == 0:
            return []

        if self.lambda_risk == 0:
            base = total_shares // intervals
            remainder = total_shares % intervals

            schedule = [base] * intervals

            for i in range(remainder):
                schedule[i] += 1

            return schedule

        weights = []

        for i in range(intervals):
            weight = 1 + self.lambda_risk * (intervals - i - 1)
            weights.append(weight)

        total_weight = sum(weights)

        schedule = []

        for weight in weights:
            shares = int(total_shares * weight / total_weight)
            schedule.append(shares)

        remainder = total_shares - sum(schedule)

        for i in range(remainder):
            schedule[i] += 1

        return schedule
