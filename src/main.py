from execution.twap import TWAPStrategy

def main():
    print("engine started")

    strategy = TWAPStrategy()
    print(strategy.generate_schedule(10, 3))
    print(strategy.generate_schedule(5, 10))
    print(strategy.generate_schedule(17, 5))

if __name__ == "__main__":
    main()
