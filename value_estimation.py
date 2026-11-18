
import random
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import pandas as pd


SEEDS = [11, 22, 33]
GAMES_PER_SEED = 10_000
CHECKPOINTS = [100, 1_000, 10_000]

GAMMA = 1.0
ALPHA = 0.1

OUTPUT_DIR = Path("outputs/simulation")


def fixed_policy(observation):
    player_sum, dealer_showing, usable_ace = observation
    return 1 if player_sum < 20 else 0


def collect_games(seed, number_of_games):
    env = gym.make(
        "Blackjack-v1",
        natural=False,
        sab=False
    )

    games = []

    for game_number in range(1, number_of_games + 1):
        observation, info = env.reset(seed=seed + game_number)

        transitions = []
        steps = 0

        while True:
            state = tuple(observation)
            action = fixed_policy(observation)

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            next_state = (
                tuple(next_observation)
                if not terminated and not truncated
                else None
            )

            transitions.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": float(reward),
                "done": terminated or truncated
            })

            observation = next_observation
            steps += 1

            if terminated or truncated or steps >= 100:
                break

        games.append(transitions)

    env.close()
    return games


def monte_carlo_values(games):
    returns_by_state = defaultdict(list)

    for game in games:
        total_return = 0.0

        for transition in reversed(game):
            total_return = (
                transition["reward"] + GAMMA * total_return
            )
            returns_by_state[transition["state"]].append(total_return)

    values = {
        state: sum(returns) / len(returns)
        for state, returns in returns_by_state.items()
    }

    return values, {
        state: len(returns)
        for state, returns in returns_by_state.items()
    }


def td_values(games):
    values = defaultdict(float)

    for game in games:
        for transition in game:
            state = transition["state"]
            reward = transition["reward"]
            next_state = transition["next_state"]

            if next_state is None:
                target = reward
            else:
                target = reward + GAMMA * values[next_state]

            values[state] += ALPHA * (target - values[state])

    return dict(values)


def build_model(games):
    counts = defaultdict(int)
    transition_counts = defaultdict(int)
    reward_sums = defaultdict(float)

    for game in games:
        for transition in game:
            state = transition["state"]
            action = transition["action"]
            next_state = transition["next_state"]
            reward = transition["reward"]

            pair = (state, action)
            key = (state, action, next_state)

            counts[pair] += 1
            transition_counts[key] += 1
            reward_sums[key] += reward

    model = {}

    for pair, total in counts.items():
        state, action = pair
        model[pair] = []

        for key, count in transition_counts.items():
            if key[:2] != pair:
                continue

            next_state = key[2]
            probability = count / total
            mean_reward = reward_sums[key] / count

            model[pair].append(
                (next_state, probability, mean_reward)
            )

    return model, counts


def model_based_values(model, tolerance=1e-6, max_sweeps=10_000):
    states = {
        pair[0]
        for pair in model
    }

    values = {state: 0.0 for state in states}

    for sweep in range(1, max_sweeps + 1):
        new_values = {}

        for state in states:
            action = fixed_policy(state)

            transitions = model.get((state, action), [])

            if not transitions:
                new_values[state] = values[state]
                continue

            value = 0.0

            for next_state, probability, reward in transitions:
                future_value = (
                    0.0
                    if next_state is None
                    else values.get(next_state, 0.0)
                )

                value += probability * (
                    reward + GAMMA * future_value
                )

            new_values[state] = value

        difference = max(
            abs(new_values[state] - values[state])
            for state in states
        )

        values = new_values

        if difference < tolerance:
            return values, sweep, True

    return values, max_sweeps, False


def save_checkpoint(seed, game_number, games):
    mc, visits = monte_carlo_values(games)
    td = td_values(games)

    model, counts = build_model(games)
    model_values, sweeps, converged = model_based_values(model)

    rows = []

    all_states = set(mc) | set(td) | set(model_values)

    for state in all_states:
        player_sum, dealer_showing, usable_ace = state

        rows.append({
            "seed": seed,
            "games": game_number,
            "player_sum": player_sum,
            "dealer_showing": dealer_showing,
            "usable_ace": usable_ace,
            "visits": visits.get(state, 0),
            "model_based": model_values.get(state),
            "monte_carlo": mc.get(state),
            "td_zero": td.get(state),
            "model_converged": converged,
            "model_sweeps": sweeps
        })

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for seed in SEEDS:
        print(f"\nCollecting games for seed {seed}...")
        games = collect_games(seed, GAMES_PER_SEED)

        for checkpoint in CHECKPOINTS:
            print(f"Evaluating checkpoint: {checkpoint} games")

            rows = save_checkpoint(
                seed,
                checkpoint,
                games[:checkpoint]
            )

            all_rows.extend(rows)

    output_file = OUTPUT_DIR / "value-estimates.csv"

    pd.DataFrame(all_rows).to_csv(
        output_file,
        index=False
    )

    print("\nValue estimation completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
