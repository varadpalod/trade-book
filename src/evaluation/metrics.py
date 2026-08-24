import numpy as np


class ExecutionMetrics:
    @staticmethod
    def implementation_shortfall(arrival_price, execution_prices, quantities):
        execution_prices = np.array(execution_prices)
        quantities = np.array(quantities)

        return float(np.sum((arrival_price - execution_prices) * quantities))

    @staticmethod
    def average_execution_price(execution_prices, quantities):
        execution_prices = np.array(execution_prices)
        quantities = np.array(quantities)

        if quantities.sum() == 0:
            return 0.0

        return float(np.sum(execution_prices * quantities) / np.sum(quantities))

    @staticmethod
    def execution_revenue(execution_prices, quantities):
        execution_prices = np.array(execution_prices)
        quantities = np.array(quantities)

        return float(np.sum(execution_prices * quantities))

    @staticmethod
    def remaining_inventory(remaining_shares):
        return remaining_shares

    @staticmethod
    def shares_sold(total_shares, remaining_shares):
        return total_shares - remaining_shares

    @staticmethod
    def total_reward(rewards):
        return float(np.sum(rewards))

    @staticmethod
    def participation_rate(total_shares, quantities):   
        quantities = np.array(quantities)

        if total_shares == 0:
            return 0.0

        return quantities / total_shares