#!/usr/bin/env python3
"""
Multi-seed ATSC V6.5 runner with extended logging.

Generates:
- raw results
- summary results
- reward time series
- AI time series
- queue time series
- throughput time series
- green duration time series
- PPO loss logs
"""

import csv
import time
from pathlib import Path
import numpy as np

import flood_sumo_integration_v65_paper_reward as atsc


SEEDS = [1, 7, 21, 42, 84]

OUT_DIR = Path("sim_output")
OUT_DIR.mkdir(exist_ok=True)

RAW_CSV = OUT_DIR / "multiseed_raw_results_extended.csv"
TIME_CSV = OUT_DIR / "multiseed_timeseries.csv"
PPO_CSV = OUT_DIR / "multiseed_ppo_logs.csv"
SUMMARY_CSV = OUT_DIR / "multiseed_summary_extended.csv"


def patch_build_route_file():
    original_build = atsc.build_route_file

    def build_route_file_fixed(label, depth, valid_routes):
        base_label = label.split("_seed")[0]
        old_scale = atsc.DEMAND_SCALE.copy()

        if base_label in old_scale:
            atsc.DEMAND_SCALE[label] = old_scale[base_label]

        result = original_build(label, depth, valid_routes)

        if label in atsc.DEMAND_SCALE:
            del atsc.DEMAND_SCALE[label]

        return result

    atsc.build_route_file = build_route_file_fixed


def run_scenario_logged(scenario, cfg_path, demand_issued, agent, transformer, seed):
    name = scenario["name"]
    depth = scenario["depth"]
    label = Path(cfg_path).stem

    trip_xml = str(OUT_DIR / f"{label}_tripinfo.xml")

    port = atsc._free_port()
    cmd = atsc._sumo_cmd(cfg_path)

    tls_ctrl = atsc.AdaptiveTLS(
        atsc.TLS_ID,
        agent,
        transformer,
        demand_issued
    )

    flood_ctrl = atsc.FloodController(atsc.FLOOD_EDGES, depth)

    tot_arrived = 0
    tot_teleport = 0
    tot_reward = 0.0

    time_rows = []
    ppo_rows = []

    print("\n" + "=" * 68)
    print(f"SCENARIO: {name} | seed={seed} | depth={depth:.2f} m | demand={demand_issued}")
    print("=" * 68)

    try:
        atsc.traci.start(cmd, port=port, numRetries=5, verbose=False)

        tls_ctrl.init()
        flood_ctrl.init()
        flood_ctrl.apply()

        for step in range(atsc.SIM_DURATION):
            atsc.traci.simulationStep()

            arr = int(atsc.traci.simulation.getArrivedNumber())
            tel = int(atsc.traci.simulation.getStartingTeleportNumber())

            tot_arrived += arr
            tot_teleport += tel

            queue = 0
            step_wait_total = 0.0

            for edge in atsc.APPROACH_EDGES:
                try:
                    queue += int(atsc.traci.edge.getLastStepHaltingNumber(edge))
                    step_wait_total += atsc.traci.edge.getWaitingTime(edge)
                except Exception:
                    pass

            active_vehicles = len(atsc.traci.vehicle.getIDList())
            avg_wait_step = step_wait_total / max(active_vehicles, 1)

            info = tls_ctrl.tick(arr, queue, depth, avg_wait_step)
            tot_reward += info["reward"]

            if step > 0 and step % 60 == 0:
                flood_ctrl.apply()

            current_ai, _ = atsc.compute_AI(
                arrived=tot_arrived,
                demand_issued=demand_issued,
                avg_wait_s=avg_wait_step
            )

            if step % 60 == 0:
                time_rows.append({
                    "seed": seed,
                    "scenario": name,
                    "depth": depth,
                    "step": step,
                    "arrived_cum": tot_arrived,
                    "arrivals_step": arr,
                    "queue": queue,
                    "avg_wait_step": avg_wait_step,
                    "reward": info["reward"],
                    "cum_reward": tot_reward,
                    "AI_step": current_ai,
                    "phase": info["phase"],
                    "green_duration": info["dur"],
                    "active_vehicles": active_vehicles,
                })

            if info["ppo"]:
                ppo_rows.append({
                    "seed": seed,
                    "scenario": name,
                    "depth": depth,
                    "step": step,
                    "actor_loss": info["ppo"].get("al", np.nan),
                    "critic_loss": info["ppo"].get("cl", np.nan),
                    "ppo_updates": agent.n_updates,
                })

            if step % 300 == 0:
                print(
                    f"t={step:4d}s | arr={tot_arrived:4d} | "
                    f"queue={queue:3d} | wait={avg_wait_step:6.2f} | "
                    f"AI={current_ai:6.3f} | R={info['reward']:7.2f}"
                )

        flood_ctrl.restore()
        atsc._safe_close()

    except Exception as exc:
        print(f"[ERROR] {name}, seed={seed}: {exc}")
        atsc._safe_close()
        return None, time_rows, ppo_rows

    avg_wait = atsc.parse_tripinfo_wait(trip_xml)

    if avg_wait == 0.0:
        print(f"[WARN] Tripinfo wait missing for {trip_xml}; using time-series estimate.")
        avg_wait = np.mean([r["avg_wait_step"] for r in time_rows]) if time_rows else 0.0

    tput = tot_arrived / (atsc.SIM_DURATION / 3600.0)
    AI, ai_label = atsc.compute_AI(tot_arrived, demand_issued, avg_wait)

    result = {
        "seed": seed,
        "scenario": name,
        "flood_depth_m": round(depth, 2),
        "demand_issued": demand_issued,
        "total_arrived": tot_arrived,
        "total_teleport": tot_teleport,
        "pct_served": round(100.0 * tot_arrived / max(demand_issued, 1), 2),
        "avg_wait_s": round(avg_wait, 3),
        "throughput_vehph": round(tput, 2),
        "AI": AI,
        "AI_label": ai_label,
        "cum_reward": round(tot_reward, 3),
        "ppo_updates": agent.n_updates,
    }

    print("RESULT:", result)

    return result, time_rows, ppo_rows


def save_csv(path, rows):
    if not rows:
        print(f"[WARN] No rows for {path}")
        return

    keys = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SAVED] {path}")


def summarize(raw_rows):
    scenario_order = ["Baseline", "Mild", "Moderate", "Severe"]
    summary = []

    for scen in scenario_order:
        rows = [r for r in raw_rows if r["scenario"] == scen]

        def mean(key):
            return float(np.mean([r[key] for r in rows]))

        def std(key):
            return float(np.std([r[key] for r in rows], ddof=1))

        summary.append({
            "scenario": scen,
            "depth": mean("flood_depth_m"),
            "AI_mean": mean("AI"),
            "AI_std": std("AI"),
            "service_mean": mean("pct_served"),
            "service_std": std("pct_served"),
            "wait_mean": mean("avg_wait_s"),
            "wait_std": std("avg_wait_s"),
            "throughput_mean": mean("throughput_vehph"),
            "throughput_std": std("throughput_vehph"),
            "reward_mean": mean("cum_reward"),
            "reward_std": std("cum_reward"),
            "ppo_updates_mean": mean("ppo_updates"),
            "ppo_updates_std": std("ppo_updates"),
        })

    return summary


def main():
    patch_build_route_file()

    raw_rows = []
    all_time_rows = []
    all_ppo_rows = []

    for seed in SEEDS:
        print("\n" + "#" * 70)
        print(f"RUNNING SEED {seed}")
        print("#" * 70)

        np.random.seed(seed)

        valid_routes = atsc.get_valid_routes()
        transformer = atsc.LightweightTransformer(seed=seed)
        agent = atsc.PPOAgent(seed=seed)

        configs = []

        for sc in atsc.SCENARIOS:
            run_label = f"{sc['label']}_seed{seed}"

            rou, n_veh = atsc.build_route_file(
                run_label,
                sc["depth"],
                valid_routes
            )

            cfg = atsc.write_sumo_cfg(run_label, rou)
            configs.append((sc, cfg, n_veh))

        for sc, cfg, n_veh in configs:
            result, time_rows, ppo_rows = run_scenario_logged(
                scenario=sc,
                cfg_path=cfg,
                demand_issued=n_veh,
                agent=agent,
                transformer=transformer,
                seed=seed
            )

            if result:
                raw_rows.append(result)

            all_time_rows.extend(time_rows)
            all_ppo_rows.extend(ppo_rows)

            time.sleep(1)

    summary_rows = summarize(raw_rows)

    save_csv(RAW_CSV, raw_rows)
    save_csv(TIME_CSV, all_time_rows)
    save_csv(PPO_CSV, all_ppo_rows)
    save_csv(SUMMARY_CSV, summary_rows)

    print("\nFINAL SUMMARY")
    print("=" * 70)
    for r in summary_rows:
        print(
            f"{r['scenario']:<10} "
            f"AI={r['AI_mean']:.4f}±{r['AI_std']:.4f} | "
            f"Service={r['service_mean']:.2f}±{r['service_std']:.2f}% | "
            f"Wait={r['wait_mean']:.2f}±{r['wait_std']:.2f}s"
        )


if __name__ == "__main__":
    main()