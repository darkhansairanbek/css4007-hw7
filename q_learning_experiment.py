
import os
import random

import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt


TRAINING_SEEDS = [11, 22, 33]
EVALUATION_SEEDS = [1011, 1022, 1033]

TRAINING_GAMES = 50_000
EVALUATION_GAMES = 10_000

ALPHA = 0.1
GAMMA = 1.0
EPSILONS = [0.2, 0.0]

MAX_PLAYER_ACTIONS = 100

OUTPUT_DIR = "outputs/q-learning"
PLOT_DIR = "plots"


def make_env():
    return gym.make(
        "Blackjack-v1",
        natural=False,
        sab=False
    )


def state_key(observation):
    player_sum, dealer_showing, usable_ace = observation

    return (
        int(player_sum),
        int(dealer_showing),
        int(usable_ace)
    )


def get_q_values(q_table, state):
    return q_table.get(state, [0.0, 0.0])


def choose_action(q_table, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice([0, 1])

    values = get_q_values(q_table, state)
    best_value = max(values)

    best_actions = [
        action for action in [0, 1]
        if values[action] == best_value
    ]

    return rng.choice(best_actions)


def choose_greedy_action(q_table, state, rng):
    values = get_q_values(q_table, state)
    best_value = max(values)

    best_actions = [
        action for action in [0, 1]
        if values[action] == best_value
    ]

    return rng.choice(best_actions)


def fixed_policy_action(state):
    player_sum, dealer_showing, usable_ace = state

    if player_sum < 20:
        return 1

    return 0


def train_q_learning(seed, epsilon):
    env = make_env()

    rng = random.Random(seed)
    env.action_space.seed(seed)

    q_table = {}
    curve_rows = []

    block_wins = 0
    block_losses = 0
    block_draws = 0
    block_cutoffs = 0
    block_reward = 0.0

    for game in range(1, TRAINING_GAMES + 1):

        if game == 1:
            observation, info = env.reset(seed=seed)
        else:
            observation, info = env.reset()

        state = state_key(observation)

        episode_reward = 0.0
        episode_cutoff = False

        for action_number in range(1, MAX_PLAYER_ACTIONS + 1):

            if state not in q_table:
                q_table[state] = [0.0, 0.0]

            action = choose_action(
                q_table,
                state,
                epsilon,
                rng
            )

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            next_state = state_key(next_observation)
            episode_reward += reward

            forced_cutoff = (
                action_number == MAX_PLAYER_ACTIONS
                and not terminated
                and not truncated
            )

            if terminated or truncated or forced_cutoff:
                target = reward
            else:
                next_values = get_q_values(q_table, next_state)
                target = reward + GAMMA * max(next_values)

            old_value = q_table[state][action]

            q_table[state][action] = (
                old_value + ALPHA * (target - old_value)
            )

            state = next_state

            if terminated:
                break

            if truncated or forced_cutoff:
                episode_cutoff = True
                episode_reward = 0.0
                break

        if episode_cutoff:
            block_cutoffs += 1
        elif episode_reward > 0:
            block_wins += 1
        elif episode_reward < 0:
            block_losses += 1
        else:
            block_draws += 1

        block_reward += episode_reward

        if game % 1000 == 0:

            curve_rows.append({
                "seed": seed,
                "epsilon": epsilon,
                "games": game,
                "wins": block_wins,
                "losses": block_losses,
                "draws": block_draws,
                "cutoffs": block_cutoffs,
                "mean_reward": block_reward / 1000
            })

            print(
                f"Seed={seed}, epsilon={epsilon}, "
                f"games={game}, "
                f"mean reward={block_reward / 1000:.4f}"
            )

            block_wins = 0
            block_losses = 0
            block_draws = 0
            block_cutoffs = 0
            block_reward = 0.0

    env.close()

    return q_table, curve_rows


def evaluate_policy(
    policy_name,
    q_table,
    epsilon,
    training_seed,
    evaluation_seed
):
    env = make_env()

    rng = random.Random(evaluation_seed + 500_000)
    env.action_space.seed(evaluation_seed)

    wins = 0
    losses = 0
    draws = 0
    cutoffs = 0
    total_reward = 0.0

    for game in range(1, EVALUATION_GAMES + 1):

        if game == 1:
            observation, info = env.reset(seed=evaluation_seed)
        else:
            observation, info = env.reset()

        state = state_key(observation)

        episode_reward = 0.0
        episode_cutoff = False

        for action_number in range(1, MAX_PLAYER_ACTIONS + 1):

            if policy_name == "fixed_baseline":
                action = fixed_policy_action(state)
            else:
                action = choose_greedy_action(
                    q_table,
                    state,
                    rng
                )

            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            state = state_key(next_observation)
            episode_reward += reward

            forced_cutoff = (
                action_number == MAX_PLAYER_ACTIONS
                and not terminated
                and not truncated
            )

            if terminated:
                break

            if truncated or forced_cutoff:
                episode_cutoff = True
                episode_reward = 0.0
                break

        total_reward += episode_reward

        if episode_cutoff:
            cutoffs += 1
        elif episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()

    return {
        "policy": policy_name,
        "epsilon": epsilon,
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "games": EVALUATION_GAMES,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "cutoffs": cutoffs,
        "mean_reward": total_reward / EVALUATION_GAMES
    }


def save_q_table(q_table, seed, epsilon, rows):
    for state, values in q_table.items():

        rows.append({
            "training_seed": seed,
            "epsilon": epsilon,
            "player_sum": state[0],
            "dealer_showing": state[1],
            "usable_ace": state[2],
            "q_stick": values[0],
            "q_hit": values[1]
        })


def create_training_plots(curves):
    os.makedirs(PLOT_DIR, exist_ok=True)

    for epsilon in EPSILONS:

        subset = curves[curves["epsilon"] == epsilon]

        averaged = (
            subset.groupby("games")
            .agg({
                "wins": "mean",
                "losses": "mean",
                "draws": "mean",
                "cutoffs": "mean",
                "mean_reward": "mean"
            })
            .reset_index()
        )

        averaged["win_rate"] = averaged["wins"] / 1000
        averaged["loss_rate"] = averaged["losses"] / 1000
        averaged["draw_rate"] = averaged["draws"] / 1000
        averaged["cutoff_rate"] = averaged["cutoffs"] / 1000

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(
            averaged["games"],
            averaged["win_rate"]
        )
        axes[0, 0].set_title("Win rate")
        axes[0, 0].set_xlabel("Training games")
        axes[0, 0].set_ylabel("Rate")

        axes[0, 1].plot(
            averaged["games"],
            averaged["loss_rate"]
        )
        axes[0, 1].set_title("Loss rate")
        axes[0, 1].set_xlabel("Training games")
        axes[0, 1].set_ylabel("Rate")

        axes[1, 0].plot(
            averaged["games"],
            averaged["draw_rate"],
            label="Draws"
        )
        axes[1, 0].plot(
            averaged["games"],
            averaged["cutoff_rate"],
            label="Cutoffs"
        )
        axes[1, 0].set_title("Draw and cutoff rates")
        axes[1, 0].set_xlabel("Training games")
        axes[1, 0].set_ylabel("Rate")
        axes[1, 0].legend()

        axes[1, 1].plot(
            averaged["games"],
            averaged["mean_reward"]
        )
        axes[1, 1].set_title("Mean reward")
        axes[1, 1].set_xlabel("Training games")
        axes[1, 1].set_ylabel("Reward")

        fig.suptitle(
            f"Q-learning training curves, epsilon={epsilon}"
        )

        fig.tight_layout()

        filename = f"q-learning-epsilon-{epsilon}.png"

        fig.savefig(
            os.path.join(PLOT_DIR, filename),
            dpi=150
        )

        plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    all_curves = []
    all_q_values = []
    all_evaluations = []

    trained_policies = []

    for epsilon in EPSILONS:

        for seed in TRAINING_SEEDS:

            print()
            print(
                f"Training Q-learning: "
                f"seed={seed}, epsilon={epsilon}"
            )

            q_table, curve_rows = train_q_learning(
                seed,
                epsilon
            )

            all_curves.extend(curve_rows)

            save_q_table(
                q_table,
                seed,
                epsilon,
                all_q_values
            )

            trained_policies.append({
                "seed": seed,
                "epsilon": epsilon,
                "q_table": q_table
            })

            print(
                f"Training finished. "
                f"Learned states: {len(q_table)}"
            )

    print()
    print("Evaluating fixed baseline...")

    for evaluation_seed in EVALUATION_SEEDS:

        result = evaluate_policy(
            "fixed_baseline",
            None,
            None,
            "baseline",
            evaluation_seed
        )

        all_evaluations.append(result)
        print(result)

    print()
    print("Evaluating trained Q-learning policies...")

    for trained in trained_policies:

        for evaluation_seed in EVALUATION_SEEDS:

            result = evaluate_policy(
                "q_learning",
                trained["q_table"],
                trained["epsilon"],
                trained["seed"],
                evaluation_seed
            )

            all_evaluations.append(result)
            print(result)

    curves_df = pd.DataFrame(all_curves)
    q_values_df = pd.DataFrame(all_q_values)
    evaluations_df = pd.DataFrame(all_evaluations)

    summary_df = (
        evaluations_df.groupby(
            ["policy", "epsilon", "training_seed"],
            dropna=False
        )
        .agg({
            "wins": "mean",
            "losses": "mean",
            "draws": "mean",
            "cutoffs": "mean",
            "mean_reward": "mean"
        })
        .reset_index()
    )

    curves_df.to_csv(
        os.path.join(OUTPUT_DIR, "training-curves.csv"),
        index=False
    )

    q_values_df.to_csv(
        os.path.join(OUTPUT_DIR, "q-tables.csv"),
        index=False
    )

    evaluations_df.to_csv(
        os.path.join(OUTPUT_DIR, "evaluation-results.csv"),
        index=False
    )

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "evaluation-summary.csv"),
        index=False
    )

    create_training_plots(curves_df)

    print()
    print("Q-learning experiment completed.")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Plots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
