
from pathlib import Path
import pandas as pd


GAMES_FILE = Path("data/paper-games.csv")
OUTPUT_DIR = Path("outputs/paper-replay")

ALPHA = 0.1
GAMMA = 1.0

STATES = ["A", "B", "C", "D", "E"]
ACTIONS = [0, 1]


def run_q_learning(games, number_of_passes=2):
    q_values = {
        state: {action: 0.0 for action in ACTIONS}
        for state in STATES
    }

    print("Q-LEARNING")
    print("=" * 55)

    for pass_number in range(1, number_of_passes + 1):
        print(f"\nPASS {pass_number}")

        for _, row in games.iterrows():
            state = row["state"]
            action = int(row["action"])
            next_state = row["next_state"]
            reward = float(row["reward"])

            old_q = q_values[state][action]

            if next_state == "END":
                max_next_q = 0.0
            else:
                max_next_q = max(q_values[next_state].values())

            target = reward + GAMMA * max_next_q

            new_q = old_q + ALPHA * (target - old_q)

            q_values[state][action] = new_q

            print(
                f"State={state}, action={action}, "
                f"next={next_state}: "
                f"old Q={old_q:.6f}, "
                f"target={target:.6f}, "
                f"new Q={new_q:.6f}"
            )

        print("\nQ-values after pass:")

        for state in STATES:
            print(
                f"{state}: "
                f"Stick={q_values[state][0]:.6f}, "
                f"Hit={q_values[state][1]:.6f}"
            )

    return q_values


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(GAMES_FILE)

    q_values = run_q_learning(games)

    output_file = OUTPUT_DIR / "q-learning.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("Q-LEARNING FINAL TABLE\n\n")

        for state in STATES:
            file.write(
                f"{state}: "
                f"Stick={q_values[state][0]:.6f}, "
                f"Hit={q_values[state][1]:.6f}\n"
            )

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
