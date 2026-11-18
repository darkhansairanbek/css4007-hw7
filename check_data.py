
from pathlib import Path
import pandas as pd


# Paths to the input files
DATA_DIR = Path("data")

STATES_FILE = DATA_DIR / "states.csv"
GAMES_FILE = DATA_DIR / "paper-games.csv"


def check_states():
    states = pd.read_csv(STATES_FILE)

    required_columns = [
        "label",
        "player_sum",
        "dealer_showing",
        "usable_ace"
    ]

    for column in required_columns:
        if column not in states.columns:
            raise ValueError(f"Missing column in states.csv: {column}")

    if states["label"].duplicated().any():
        raise ValueError("State labels must be unique.")

    if not states["usable_ace"].isin([0, 1]).all():
        raise ValueError("usable_ace must contain only 0 or 1.")

    print("States file: OK")
    print(f"Number of states: {len(states)}")

    return states


def check_games(states):
    games = pd.read_csv(GAMES_FILE)

    required_columns = [
        "game",
        "step",
        "state",
        "action",
        "next_state",
        "reward",
        "terminated"
    ]

    for column in required_columns:
        if column not in games.columns:
            raise ValueError(f"Missing column in paper-games.csv: {column}")

    known_states = set(states["label"])
    
    if not games["state"].isin(known_states).all():
        raise ValueError("Unknown state found in the state column.")

    valid_next_states = known_states | {"END"}

    if not games["next_state"].isin(valid_next_states).all():
        raise ValueError("Unknown next_state found.")

    if not games["action"].isin([0, 1]).all():
        raise ValueError("Actions must be 0 (stick) or 1 (hit).")

    if not games["reward"].isin([-1, 0, 1]).all():
        raise ValueError("Rewards must be -1, 0 or 1.")

    terminated = (
        games["terminated"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

    if terminated.isna().any():
        raise ValueError("Invalid terminated values.")

    if not (games["next_state"].eq("END") == terminated).all():
        raise ValueError(
            "END transitions and terminated values do not match."
        )

    for game_id, group in games.groupby("game"):
        steps = group["step"].tolist()

        expected_steps = list(range(1, len(steps) + 1))

        if steps != expected_steps:
            raise ValueError(f"Incorrect step sequence in game {game_id}.")

        if group.iloc[-1]["next_state"] != "END":
            raise ValueError(f"Game {game_id} does not end with END.")

    print("Paper games file: OK")
    print(f"Number of games: {games['game'].nunique()}")
    print(f"Number of transitions: {len(games)}")

    return games


def main():
    print("Checking HW7 input data...\n")

    states = check_states()
    games = check_games(states)

    print("\nAll input data checks passed.")


if __name__ == "__main__":
    main()
