
import random
from pathlib import Path

import gymnasium as gym
import pandas as pd


SEEDS = [11, 22, 33]
GAMES_PER_SEED = 10_000
CHECKPOINTS = [100, 1_000, 10_000]

OUTPUT_DIR = Path("outputs/simulation")


def fixed_policy(observation):
    player_sum, dealer_showing, usable_ace = observation

    if player_sum < 20:
        return 1

    return 0


def run_experiment(seed, number_of_games):
    env = gym.make(
        "Blackjack-v1",
        natural=False,
        sab=False
    )

    random.seed(seed)

    state_visits = {}
    rewards = []
    results = []

    for game_number in range(1, number_of_games + 1):
        observation, info = env.reset(seed=seed + game_number)

        episode_reward = 0
        steps = 0

        while True:
            action = fixed_policy(observation)

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            state = tuple(observation)
            state_visits[state] = state_visits.get(state, 0) + 1

            episode_reward += reward
            steps += 1
            observation = next_observation

            if terminated or truncated or steps >= 100:
                break

        rewards.append(episode_reward)

        if game_number in CHECKPOINTS:
            results.append({
                "seed": seed,
                "games": game_number,
                "mean_reward": sum(rewards) / len(rewards),
                "states_visited": len(state_visits),
                "total_visits": sum(state_visits.values())
            })

    env.close()

    return results, state_visits, rewards


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_visits = []

    for seed in SEEDS:
        print(f"\nRunning fixed-policy experiment, seed={seed}")

        results, visits, rewards = run_experiment(
            seed,
            GAMES_PER_SEED
        )

        all_results.extend(results)

        for state, count in visits.items():
            player_sum, dealer_showing, usable_ace = state

            all_visits.append({
                "seed": seed,
                "player_sum": player_sum,
                "dealer_showing": dealer_showing,
                "usable_ace": usable_ace,
                "visits": count
            })

        for result in results:
            print(result)

    pd.DataFrame(all_results).to_csv(
        OUTPUT_DIR / "fixed-policy-checkpoints.csv",
        index=False
    )

    pd.DataFrame(all_visits).to_csv(
        OUTPUT_DIR / "fixed-policy-state-visits.csv",
        index=False
    )

    print("\nFixed-policy experiment completed.")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
