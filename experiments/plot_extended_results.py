#!/usr/bin/env python3
"""
Generate extended publication-quality figures from ATSC V6.5 multi-seed logs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


OUT_DIR = Path("sim_output")
FIG_DIR = OUT_DIR / "figures_extended"
FIG_DIR.mkdir(exist_ok=True)

RAW_CSV = OUT_DIR / "multiseed_raw_results_extended.csv"
TIME_CSV = OUT_DIR / "multiseed_timeseries.csv"
PPO_CSV = OUT_DIR / "multiseed_ppo_logs.csv"
SUMMARY_CSV = OUT_DIR / "multiseed_summary_extended.csv"

SCENARIO_ORDER = ["Baseline", "Mild", "Moderate", "Severe"]


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def savefig(name):
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {png_path}")
    print(f"[SAVED] {pdf_path}")


def ordered_summary(summary):
    summary["scenario"] = pd.Categorical(
        summary["scenario"],
        categories=SCENARIO_ORDER,
        ordered=True
    )
    return summary.sort_values("scenario")


def plot_ai_vs_depth(summary):
    plt.figure(figsize=(6.5, 4.2))

    plt.errorbar(
        summary["depth"],
        summary["AI_mean"],
        yerr=summary["AI_std"],
        marker="o",
        linewidth=2,
        capsize=5,
        label="Mean AI ± SD"
    )

    plt.axhline(0.70, linestyle="--", linewidth=1.5, label="Antifragile threshold")
    plt.axhline(0.50, linestyle="--", linewidth=1.5, label="Resilient threshold")

    plt.xlabel("Flood depth (m)")
    plt.ylabel("Antifragility Index")
    plt.title("Antifragility Index Across Flood Depths")
    plt.legend(loc="best")
    savefig("fig_ai_vs_depth")


def plot_service_rate(summary):
    x = np.arange(len(summary))

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x, summary["service_mean"], yerr=summary["service_std"], capsize=5)

    plt.xticks(x, summary["scenario"])
    plt.ylabel("Service rate (%)")
    plt.title("Vehicle Service Rate Across Flood Scenarios")
    plt.ylim(0, 105)
    savefig("fig_service_rate")


def plot_waiting_time(summary):
    x = np.arange(len(summary))

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x, summary["wait_mean"], yerr=summary["wait_std"], capsize=5)

    plt.xticks(x, summary["scenario"])
    plt.ylabel("Average waiting time (s)")
    plt.title("Average Waiting Time Across Flood Scenarios")
    savefig("fig_waiting_time")


def plot_throughput(summary):
    x = np.arange(len(summary))

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x, summary["throughput_mean"], yerr=summary["throughput_std"], capsize=5)

    plt.xticks(x, summary["scenario"])
    plt.ylabel("Throughput (veh/h)")
    plt.title("Throughput Across Flood Scenarios")
    savefig("fig_throughput")


def plot_reward_by_scenario(summary):
    x = np.arange(len(summary))

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x, summary["reward_mean"], yerr=summary["reward_std"], capsize=5)

    plt.xticks(x, summary["scenario"])
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative Reward Across Flood Scenarios")
    savefig("fig_cumulative_reward")


def plot_ppo_updates(summary):
    x = np.arange(len(summary))

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x, summary["ppo_updates_mean"])

    plt.xticks(x, summary["scenario"])
    plt.ylabel("Cumulative PPO updates")
    plt.title("PPO Learning Updates Across Scenarios")
    savefig("fig_ppo_updates")


def plot_ai_evolution(timeseries):
    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = timeseries[timeseries["scenario"] == scen]
        grouped = df.groupby("step")["AI_step"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.axhline(0.70, linestyle="--", linewidth=1.2)
    plt.axhline(0.50, linestyle="--", linewidth=1.2)

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Step-level AI")
    plt.title("Temporal Evolution of Antifragility Index")
    plt.legend()
    savefig("fig_ai_evolution_over_time")


def plot_reward_evolution(timeseries):
    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = timeseries[timeseries["scenario"] == scen]
        grouped = df.groupby("step")["cum_reward"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative Reward Evolution")
    plt.legend()
    savefig("fig_reward_evolution_over_time")


def plot_queue_evolution(timeseries):
    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = timeseries[timeseries["scenario"] == scen]
        grouped = df.groupby("step")["queue"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Queue length / halting vehicles")
    plt.title("Queue Evolution Over Time")
    plt.legend()
    savefig("fig_queue_evolution_over_time")


def plot_throughput_evolution(timeseries):
    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = timeseries[timeseries["scenario"] == scen].copy()

        df["veh_per_min"] = df.groupby(["seed"])["arrived_cum"].diff().fillna(df["arrived_cum"])
        grouped = df.groupby("step")["veh_per_min"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Vehicles served per 60 s")
    plt.title("Throughput Evolution Over Time")
    plt.legend()
    savefig("fig_throughput_evolution_over_time")


def plot_green_duration(timeseries):
    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = timeseries[timeseries["scenario"] == scen]
        grouped = df.groupby("step")["green_duration"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Green duration (s)")
    plt.title("Adaptive Green-Time Evolution")
    plt.legend()
    savefig("fig_green_time_evolution")


def plot_actor_loss(ppo):
    if ppo.empty:
        print("[WARN] PPO log is empty; skipping actor loss.")
        return

    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = ppo[ppo["scenario"] == scen]
        grouped = df.groupby("step")["actor_loss"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Actor loss")
    plt.title("PPO Actor Loss")
    plt.legend()
    savefig("fig_actor_loss")


def plot_critic_loss(ppo):
    if ppo.empty:
        print("[WARN] PPO log is empty; skipping critic loss.")
        return

    plt.figure(figsize=(7, 4.5))

    for scen in SCENARIO_ORDER:
        df = ppo[ppo["scenario"] == scen]
        grouped = df.groupby("step")["critic_loss"].agg(["mean", "std"]).reset_index()

        plt.plot(grouped["step"], grouped["mean"], linewidth=2, label=scen)
        plt.fill_between(
            grouped["step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15
        )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Critic loss")
    plt.title("PPO Critic Loss")
    plt.legend()
    savefig("fig_critic_loss")


def plot_waiting_time_boxplot(raw):
    data = [raw[raw["scenario"] == scen]["avg_wait_s"].values for scen in SCENARIO_ORDER]

    plt.figure(figsize=(6.5, 4.2))
    plt.boxplot(data, labels=SCENARIO_ORDER, showmeans=True)

    plt.ylabel("Average waiting time (s)")
    plt.title("Distribution of Waiting Time Across Seeds")
    savefig("fig_waiting_time_boxplot")


def plot_ai_boxplot(raw):
    data = [raw[raw["scenario"] == scen]["AI"].values for scen in SCENARIO_ORDER]

    plt.figure(figsize=(6.5, 4.2))
    plt.boxplot(data, labels=SCENARIO_ORDER, showmeans=True)

    plt.axhline(0.70, linestyle="--", linewidth=1.2, label="Antifragile threshold")
    plt.axhline(0.50, linestyle="--", linewidth=1.2, label="Resilient threshold")

    plt.ylabel("Antifragility Index")
    plt.title("Distribution of AI Across Seeds")
    plt.legend()
    savefig("fig_ai_boxplot")


def main():
    apply_style()

    raw = pd.read_csv(RAW_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    timeseries = pd.read_csv(TIME_CSV)

    if PPO_CSV.exists():
        ppo = pd.read_csv(PPO_CSV)
    else:
        ppo = pd.DataFrame()

    summary = ordered_summary(summary)

    plot_ai_vs_depth(summary)
    plot_service_rate(summary)
    plot_waiting_time(summary)
    plot_throughput(summary)
    plot_reward_by_scenario(summary)
    plot_ppo_updates(summary)

    plot_ai_evolution(timeseries)
    plot_reward_evolution(timeseries)
    plot_queue_evolution(timeseries)
    plot_throughput_evolution(timeseries)
    plot_green_duration(timeseries)

    plot_actor_loss(ppo)
    plot_critic_loss(ppo)

    plot_waiting_time_boxplot(raw)
    plot_ai_boxplot(raw)

    print("\nAll figures saved to:")
    print(FIG_DIR)


if __name__ == "__main__":
    main()