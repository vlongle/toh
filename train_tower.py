from stable_baselines3 import PPO
from tower_of_hanoi import TowerOfHanoi
from gym.wrappers import TimeLimit
from itertools import product
from rich.table import Table
from rich.console import Console
import torch


def print_V_values(model, env):
    with torch.no_grad():
        all_states = list(product(range(3), repeat=env.num_discs))
        all_states = torch.tensor(
            all_states, dtype=torch.float32).to(model.device)
        values = model.policy.predict_values(all_states)
        # put in table format
        table = Table(title="V(s) values")
        table.add_column("State", justify="center", style="cyan")
        table.add_column("Value", justify="center", style="magenta")
        for state, value in zip(all_states, values):
            table.add_row(str(state.tolist()), f"{value.item():.4f}")
        console = Console()
        console.print(table)


if __name__ == "__main__":

    NUM_DISCS = 2
    TIME_LIMIT = 40
    NUM_EPISODES = 200

    # NUM_DISCS = 1
    # TIME_LIMIT = 10
    # NUM_EPISODES = 100

    INVALID_MOVE_PENALTY = -0.01
    # INVALID_MOVE_PENALTY = -1.0

    env = TimeLimit(TowerOfHanoi(num_discs=NUM_DISCS, INVALID_MOVE_PENALTY=INVALID_MOVE_PENALTY),
                    max_episode_steps=TIME_LIMIT)
    model = PPO("MlpPolicy", env, verbose=1,  device="cuda",
                n_steps=min(2048, TIME_LIMIT),
                seed=0,
                gamma=1.0,
                )

    print("before")
    print_V_values(model, env)
    model.learn(total_timesteps=TIME_LIMIT * NUM_EPISODES, log_interval=20)

    print("after")
    print_V_values(model, env)
