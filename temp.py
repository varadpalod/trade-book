from src.execution.discrete_optimizer import DiscreteOptimizer

optimizer = DiscreteOptimizer()

for lam in [0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5]:
    schedule = optimizer.optimize(
        total_shares=1000,
        intervals=5,
        execution_horizon=60,
        sigma=0.02,
        eta=2.5e-6,
        gamma=2.5e-7,
        lambda_risk=lam,
    )
    equal = [200, 200, 200, 200, 200]
    front = [300, 250, 200, 150, 100]


    

    print(lam, schedule)