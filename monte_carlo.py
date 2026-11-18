
from pathlib import Path
from collections import defaultdict
import pandas as pd


GAMES_FILE = Path("data/paper-games.csv")
OUTPUT_DIR = Path("outputs/paper-replay")

GAMMA = 1.0


def calculate_returns(games):
    returns = defaultdict(list)

    print("EVERY-VISIT MONTE CARLO")
    print("=" * 45)

    for game_id, game in games.groupby("game", sort=True):
        game = game.sort_values("step").reset_index(drop=True)

        rewards = game["reward"].astype(float).tolist()
        states = game["state"].tolist()

        game_returns = [0.0] * len(rewards)
        G = 0.0

        for i in range(len(rewards) - 1, -1, -1):
            G = rewards[i] + GAMMA * G
            game_returns[i] = G

        print(f"\nGame {game_id}")

        for state, G in zip(states, game_returns):
            returns[state].append(G)
            print(f"State {state}: return = {G:.4f}")

    print("\nFINAL MONTE CARLO VALUES")

    results = {}

    for state in ["A", "B", "C", "D", "E"]:
        state_returns = returns[state]

        if state_returns:
            average = sum(state_returns) / len(state_returns)
            count = len(state_returns)
        else:
            average = 0.0
            count = 0

        results[state] = (count, average)

        print(
            f"V({state}) = {average:.6f}, "
            f"visits = {count}, "
            f"returns = {state_returns}"
        )

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(GAMES_FILE)
    results = calculate_returns(games)

    output_file = OUTPUT_DIR / "monte-carlo.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("EVERY-VISIT MONTE CARLO\n\n")

        for state, (count, average) in results.items():
            file.write(
                f"{state}: visits={count}, "
                f"average_return={average:.6f}\n"
            )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
