#!/usr/bin/env python3
"""
Multi-seed ATSC V6.5 experiment runner

Runs:
  5 seeds × 4 flood scenarios

Outputs:
  sim_output/multiseed_raw_results.csv
  sim_output/multiseed_summary.csv
  sim_output/fig_ai_vs_depth.png
  sim_output/fig_service_rate.png
  sim_output/fig_waiting_time.png
  sim_output/fig_ppo_updates.png
"""

import json
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your working single-run script
import flood_sumo_integration_v65_paper_reward as atsc


SEEDS = [1, 7, 21, 42, 84]

OUT_DIR = Path("sim_output")
OUT_DIR.mkdir(exist_ok=True)

RAW_CSV = OUT_DIR / "multiseed_raw_results.csv"
SUMMARY_CSV = OUT_DIR / "multiseed_summary.csv"


def run_all_seeds():
    all_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"RUNNING FULL EXPERIMENT FOR SEED {seed}")
        print("=" * 70)

        np.random.seed(seed)

        valid_routes = atsc.get_valid_routes()

        transformer = atsc.LightweightTransformer(seed=seed)
        agent = atsc.PPOAgent(seed=seed)

        configs = []
        for sc in atsc.SCENARIOS:
            # Make route generation seed-dependent
            rou, n_veh = atsc.build_route_file(
                sc["label"] + f"_seed{seed}",
                sc["depth"],
                valid_routes
            )
            cfg = atsc.write_sumo_cfg(
                sc["label"] + f"_seed{seed}",
                rou
            )
            configs.append((sc, cfg, n_veh))

        for sc, cfg, n_veh in configs:
            result = atsc.run_scenario(
                scenario=sc,
                cfg_path=cfg,
                demand_issued=n_veh,
                agent=agent,
                transformer=transformer
            )

            if "error" not in result:
                result["seed"] = seed
                all_rows.append(result)

            time.sleep(1)

    return all_rows


def save_raw_results(rows):
    keys = [
        "seed",
        "scenario",
        "flood_depth_m",
        "demand_issued",
        "total_arrived",
        "total_teleport",
        "pct_served",
        "avg_wait_s",
        "throughput_vehph",
        "AI",
        "AI_label",
        "cum_reward",
        "ppo_updates",
    ]

    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] {RAW_CSV}")


def summarize(rows):
    scenarios = sorted(
        set(r["scenario"] for r in rows),
        key=lambda x: ["Baseline", "Mild", "Moderate", "Severe"].index(x)
    )

    summary_rows = []

    for scen in scenarios:
        rs = [r for r in rows if r["scenario"] == scen]

        row = {
            "scenario": scen,
            "depth": np.mean([r["flood_depth_m"] for r in rs]),

            "AI_mean": np.mean([r["AI"] for r in rs]),
            "AI_std": np.std([r["AI"] for r in rs], ddof=1),

            "service_mean": np.mean([r["pct_served"] for r in rs]),
            "service_std": np.std([r["pct_served"] for r in rs], ddof=1),

            "wait_mean": np.mean([r["avg_wait_s"] for r in rs]),
            "wait_std": np.std([r["avg_wait_s"] for r in rs], ddof=1),

            "throughput_mean": np.mean([r["throughput_vehph"] for r in rs]),
            "throughput_std": np.std([r["throughput_vehph"] for r in rs], ddof=1),

            "reward_mean": np.mean([r["cum_reward"] for r in rs]),
            "reward_std": np.std([r["cum_reward"] for r in rs], ddof=1),

            "ppo_updates_mean": np.mean([r["ppo_updates"] for r in rs]),
            "ppo_updates_std": np.std([r["ppo_updates"] for r in rs], ddof=1),
        }

        summary_rows.append(row)

    keys = list(summary_rows[0].keys())

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[SAVED] {SUMMARY_CSV}")

    return summary_rows


def plot_results(summary):
    labels = [r["scenario"] for r in summary]
    depths = [r["depth"] for r in summary]

    # AI vs flood depth
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        depths,
        [r["AI_mean"] for r in summary],
        yerr=[r["AI_std"] for r in summary],
        marker="o",
        capsize=5
    )
    plt.axhline(0.70, linestyle="--", linewidth=1, label="Antifragile threshold")
    plt.axhline(0.50, linestyle="--", linewidth=1, label="Resilient threshold")
    plt.xlabel("Flood depth (m)")
    plt.ylabel("Antifragility Index")
    plt.title("AI vs Flood Depth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_ai_vs_depth.png", dpi=300)
    plt.close()

    # Service rate
    plt.figure(figsize=(6, 4))
    plt.bar(labels, [r["service_mean"] for r in summary])
    plt.errorbar(
        labels,
        [r["service_mean"] for r in summary],
        yerr=[r["service_std"] for r in summary],
        fmt="none",
        capsize=5
    )
    plt.ylabel("Service rate (%)")
    plt.title("Vehicle Service Rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_service_rate.png", dpi=300)
    plt.close()

    # Waiting time
    plt.figure(figsize=(6, 4))
    plt.bar(labels, [r["wait_mean"] for r in summary])
    plt.errorbar(
        labels,
        [r["wait_mean"] for r in summary],
        yerr=[r["wait_std"] for r in summary],
        fmt="none",
        capsize=5
    )
    plt.ylabel("Average waiting time (s)")
    plt.title("Average Waiting Time")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_waiting_time.png", dpi=300)
    plt.close()

    # PPO updates
    plt.figure(figsize=(6, 4))
    plt.bar(labels, [r["ppo_updates_mean"] for r in summary])
    plt.errorbar(
        labels,
        [r["ppo_updates_mean"] for r in summary],
        yerr=[r["ppo_updates_std"] for r in summary],
        fmt="none",
        capsize=5
    )
    plt.ylabel("Cumulative PPO updates")
    plt.title("PPO Learning Updates")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_ppo_updates.png", dpi=300)
    plt.close()

    print("[SAVED] Figures saved in sim_output/")


def main():
    rows = run_all_seeds()
    save_raw_results(rows)
    summary = summarize(rows)
    plot_results(summary)

    print("\nFINAL MULTI-SEED SUMMARY")
    print("=" * 70)
    for r in summary:
        print(
            f"{r['scenario']:<10} "
            f"AI={r['AI_mean']:.4f}±{r['AI_std']:.4f} | "
            f"Service={r['service_mean']:.2f}±{r['service_std']:.2f}% | "
            f"Wait={r['wait_mean']:.2f}±{r['wait_std']:.2f}s"
        )


if __name__ == "__main__":
    main()