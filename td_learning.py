
from pathlib import Path
import pandas as pd


GAMES_FILE = Path("data/paper-games.csv")
OUTPUT_DIR = Path("outputs/paper-replay")

ALPHA = 0.1
GAMMA = 1.0

STATES = ["A", "B", "C", "D", "E"]


def run_td(games, number_of_passes=2):
    values = {state: 0.0 for state in STATES}

    print("TD(0) LEARNING")
    print("=" * 50)

    for pass_number in range(1, number_of_passes + 1):
        print(f"\nPASS {pass_number}")

        for _, row in games.iterrows():
            state = row["state"]
            next_state = row["next_state"]
            reward = float(row["reward"])

            old_value = values[state]

            if next_state == "END":
                next_value = 0.0
            else:
                next_value = values[next_state]

            target = reward + GAMMA * next_value

            new_value = old_value + ALPHA * (
                target - old_value
            )

            values[state] = new_value

            print(
                f"{state} -> {next_state}: "
                f"old={old_value:.6f}, "
                f"target={target:.6f}, "
                f"new={new_value:.6f}"
            )

        print("\nValues after pass:")

        for state in STATES:
            print(f"V({state}) = {values[state]:.6f}")

    return values


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(GAMES_FILE)

    final_values = run_td(games)

    output_file = OUTPUT_DIR / "td-learning.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("TD(0) FINAL VALUES\n\n")

        for state, value in final_values.items():
            file.write(f"V({state}) = {value:.6f}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
