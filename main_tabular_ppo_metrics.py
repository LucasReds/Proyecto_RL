import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from Environments.SimpleEnvs.EscapeRoomEnvExtended import EscapeRoomEnvExtended


# CONFIGURATION
ppo_params = dict(
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=512,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.9,
    clip_range=0.1,
    normalize_advantage=True,
    device="cpu"
)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# UTILITY FUNCTIONS
def make_monitor_folder(name):
    folder = f"logs/{name}"
    os.makedirs(folder, exist_ok=True)
    return folder


def train_from_scratch(name, env):
    folder = make_monitor_folder(name)
    env_m = Monitor(env, folder)

    model = PPO("MlpPolicy", env_m, verbose=0, **ppo_params)
    model.learn(300_000)
    model.save(f"models/ppo_{name}")

    return f"{folder}/monitor.csv"


def load_monitor(path, max_steps=None):
    df = pd.read_csv(path, skiprows=1)

    # Extra metrics
    df["reward_ma30"] = df["r"].rolling(window=30).mean()
    df["reward_ewma"] = df["r"].ewm(alpha=0.05).mean()
    if max_steps is not None:
        df["success"] = (df["l"] < max_steps).astype(int)
    else:
        df["success"] = 0
    df["success_rate10"] = df["success"].rolling(window=10).mean()
    df["reward_per_step"] = df["r"] / df["l"]

    return df

def compute_summary_metrics(df, max_steps):
    metrics = {}
    success_indices = df.index[df["success"] == 1].tolist()
    metrics["primer_ep_resuelto"] = success_indices[0] if success_indices else None
    conv = df.index[df["success_rate10"] >= 0.9].tolist()
    metrics["ep_convergencia"] = conv[0] if conv else None
    metrics["steps_promedio"] = df["l"].mean()
    metrics["episodios_truncados"] = (df["l"] >= max_steps).sum()

    return metrics


# PLOTTING
def plot_metrics(df, name):
    plt.figure(figsize=(12, 5))
    plt.plot(df["t"], df["r"], alpha=0.4, label="Reward")
    plt.plot(df["t"], df["reward_ma30"], label="Reward MA(30)")
    plt.plot(df["t"], df["reward_ewma"], label="Reward EWMA")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(f"{name} — Rewards")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_reward.png")

    plt.figure(figsize=(12, 5))
    plt.plot(df["t"], df["l"], alpha=0.4, label="Episode Length")
    plt.xlabel("Timesteps")
    plt.ylabel("Length")
    plt.title(f"{name} — Episode Length")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_length.png")

    plt.figure(figsize=(12, 5))
    plt.plot(df["t"], df["success_rate10"], label="Success Rate MA(10)")
    plt.xlabel("Timesteps")
    plt.ylabel("Success Rate")
    plt.title(f"{name} — Success Rate")
    plt.grid()
    plt.legend()
    plt.savefig(f"{name}_success.png")


def plot_compare(csv_scratch, csv_transfer, h, w, title):
    max_steps = h * w * 3

    df_s = load_monitor(csv_scratch, max_steps)
    df_t = load_monitor(csv_transfer, max_steps)
    metrics_s = compute_summary_metrics(df_s, max_steps)
    metrics_t = compute_summary_metrics(df_t, max_steps)
    print(f"{title} Scratch:", metrics_s)
    print(f"{title} Transfer:", metrics_t)

    plt.figure(figsize=(10, 5))
    plt.plot(df_s["t"], df_s["reward_ewma"], label="Scratch EWMA")
    plt.plot(df_t["t"], df_t["reward_ewma"], label="Transfer EWMA")
    plt.grid()
    plt.xlabel("Timesteps")
    plt.ylabel("Smoothed Reward")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}_compare.png")



# WEIGHT TRANSFER
def transfer_weights(model_small, model_large):
    print("\n TRANSFERRING WEIGHTS ")
    small_state = model_small.policy.state_dict()
    large_state = model_large.policy.state_dict()

    for name, param in small_state.items():
        if name in large_state and large_state[name].shape == param.shape:
            large_state[name] = param
            print("Loaded:", name)
        else:
            print("Skipped:", name)

    model_large.policy.load_state_dict(large_state)


# EXPERIMENTS
def run_experiments():

    #  Small base model 
    print("\n Training SMALL 6x6 ")
    csv_small = train_from_scratch(
        "small",
        EscapeRoomEnvExtended(6, 6, randomize_layout=False)
    )

    model_small = PPO.load("models/ppo_small")
    '''
    #  Size scaling 
    sizes = [20, 50, 100, 150, 200, 500, 1000]
    scratch_csv = []
    transfer_csv = []
    
    for s in sizes:
        print(f"\n SIZE TEST: {s}x{s} ")

        # Scratch
        scratch_path = train_from_scratch(
            f"scratch_{s}",
            EscapeRoomEnvExtended(s, s, randomize_layout=False)
        )
        scratch_csv.append(scratch_path)

        # Transfer
        folder = make_monitor_folder(f"transfer_{s}")
        env_t = Monitor(
            EscapeRoomEnvExtended(s, s, randomize_layout=False),
            folder
        )

        model_t = PPO("MlpPolicy", env_t, verbose=0, **ppo_params)
        transfer_weights(model_small, model_t)
        model_t.learn(300_000)
        model_t.save(f"models/ppo_transfer_{s}")

        transfer_csv.append(f"logs/transfer_{s}/monitor.csv")
    '''
    #  Hard environments 
    hard_conditions = [
        ("random_layout_30", dict(randomize_layout=True)),
        ("walls_20", dict(n_walls=20)),
        ("walls_50", dict(n_walls=50)),
        ("walls_100", dict(n_walls=100)),
        ("walls_200", dict(n_walls=200))
    ]

    hard_scratch = []
    hard_transfer = []

    for name, params in hard_conditions:
        print(f"\n HARD TEST: {name} ")

        h, w = 40, 40

        # Scratch
        p = train_from_scratch(
            f"{name}_scratch",
            EscapeRoomEnvExtended(h, w, **params)
        )
        hard_scratch.append(p)

        # Transfer
        folder = make_monitor_folder(f"{name}_transfer")
        env_t = Monitor(EscapeRoomEnvExtended(h, w, **params), folder)

        model_t = PPO("MlpPolicy", env_t, verbose=0, **ppo_params)
        transfer_weights(model_small, model_t)
        model_t.learn(300_000)
        model_t.save(f"models/ppo_{name}_transfer")

        hard_transfer.append(f"{folder}/monitor.csv")

    #  Generate plots 
    print("\n GENERATING PLOTS ")

    #for s, cs, ct in zip(sizes, scratch_csv, transfer_csv):
    #    plot_compare(cs, ct, s, s, f"Transfer Learning — Size {s}x{s}")

    for (name, _), cs, ct in zip(hard_conditions, hard_scratch, hard_transfer):
        plot_compare(cs, ct, 40, 40, f"Transfer Learning — {name}_40")


# Run everything
if __name__ == "__main__":
    run_experiments()
