
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd

from model_based import build_model, policy_evaluation
from monte_carlo import calculate_returns
from td_learning import run_td
from q_learning import run_q_learning


GAMES_FILE = Path("data/paper-games.csv")
OUTPUT_DIR = Path("outputs/paper-replay")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games = pd.read_csv(GAMES_FILE)

    output = StringIO()

    with redirect_stdout(output):
        print("HW7 PART C — COMPLETE PAPER REPLAY")
        print("=" * 60)

        print("\n1. MODEL-BASED LEARNING")
        model = build_model(games)
        policy_evaluation(model, number_of_sweeps=2)

        print("\n2. EVERY-VISIT MONTE CARLO")
        calculate_returns(games)

        print("\n3. TD(0)")
        run_td(games, number_of_passes=2)

        print("\n4. Q-LEARNING")
        run_q_learning(games, number_of_passes=2)

    result = output.getvalue()

    print(result)

    output_file = OUTPUT_DIR / "complete-paper-replay.txt"

    output_file.write_text(result, encoding="utf-8")

    print(f"\nComplete report saved to: {output_file}")


if __name__ == "__main__":
    main()
