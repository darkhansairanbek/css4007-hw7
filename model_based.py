
from pathlib import Path
from collections import defaultdict
import pandas as pd


GAMES_FILE = Path("data/paper-games.csv")
OUTPUT_DIR = Path("outputs/paper-replay")

GAMMA = 1.0

POLICY = {
    "A": 1,
    "B": 0,
    "C": 1,
    "D": 1,
    "E": 0
}


def build_model(games):
    counts = defaultdict(int)
    reward_sums = defaultdict(float)
    transition_counts = defaultdict(int)

    for _, row in games.iterrows():
        state = row["state"]
        action = int(row["action"])
        next_state = row["next_state"]
        reward = float(row["reward"])

        pair = (state, action)
        transition = (state, action, next_state)

        counts[pair] += 1
        transition_counts[transition] += 1
        reward_sums[transition] += reward

    model = {}

    print("OBSERVED STATE-ACTION COUNTS")

    for pair, count in sorted(counts.items()):
        print(f"{pair}: {count}")

    print("\nTRANSITION MODEL")

    for transition, count in sorted(transition_counts.items()):
        state, action, next_state = transition

        probability = count / counts[(state, action)]
        mean_reward = reward_sums[transition] / count

        model[transition] = (probability, mean_reward)

        print(
            f"{state}, action={action}, next={next_state}: "
            f"count={count}, "
            f"P={probability:.4f}, "
            f"mean reward={mean_reward:.4f}"
        )

    return model


def policy_evaluation(model, number_of_sweeps=2):
    states = ["A", "B", "C", "D", "E"]

    values = {state: 0.0 for state in states}

    print("\nSYNCHRONOUS VALUE ITERATIONS")

    for sweep in range(1, number_of_sweeps + 1):
        new_values = {}

        for state in states:
            action = POLICY[state]
            value = 0.0

            for (s, a, next_state), (probability, reward) in model.items():
                if s != state or a != action:
                    continue

                next_value = (
                    0.0 if next_state == "END"
                    else values[next_state]
                )

                value += probability * (
                    reward + GAMMA * next_value
                )

            new_values[state] = value

        values = new_values

        print(f"\nV{sweep}")

        for state in states:
            print(f"V({state}) = {values[state]:.6f}")

    return values


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(GAMES_FILE)

    model = build_model(games)

    final_values = policy_evaluation(model)

    output_file = OUTPUT_DIR / "model-based.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("MODEL-BASED LEARNING\n\n")

        for state, value in final_values.items():
            file.write(f"V({state}) = {value:.6f}\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
