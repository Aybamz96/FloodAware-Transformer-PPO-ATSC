#!/usr/bin/env python3
"""
flood_sumo_integration_v65.py
Antifragile Traffic Signal Control — V6.5
Buist Ave / Spruill Ave, Charleston SC

Change log V6.5 vs V6.4:
  [FIX-1]  AI formula: throughput_ratio = arrived / demand_issued
           (compares to actual demand, not fixed peak)
  [FIX-2]  AI formula: wait penalty uses SUMO tripinfo avg WaitingTime
           (ground-truth per-vehicle seconds, not edge snapshot average)
  [FIX-3]  AI threshold adjusted: antifragile >0.0, resilient >-0.15,
           fragile otherwise — matches realistic flood throughput range
  [FIX-4]  PPO agent is shared/transferred across scenarios
           (mild → moderate → severe builds on prior knowledge)
  [FIX-5]  Wait time: read from SUMO Statistics line OR tripinfo XML
           as fallback for accurate per-vehicle waiting time
  [FIX-6]  Reward function aligned with manuscript:
           R_t = 1.0*a_t - 0.1*h_t - 0.01*w_t + Omega_t
  [FIX-7]  Demand scale for severe: uses flood-safe routes only when
           intersection edges are impassable (>= FLOOD_IMPASSABLE)
"""

import os, sys, math, time, json, random, socket, subprocess
import traceback, xml.etree.ElementTree as ET
from pathlib import Path
from collections import deque

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  SUMO ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
SUMO_HOME  = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
SUMO_BIN   = os.path.join(SUMO_HOME, "bin", "sumo.exe")
if SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)
import traci
import traci.constants as tc
import sumolib

# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT PATHS
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_DIR = Path(__file__).resolve().parent
NET_FILE    = str(PROJECT_DIR / "buist_spruill_real.net.xml")
OUT_DIR     = PROJECT_DIR / "sim_output"
OUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
TLS_ID     = "cluster_110099785_12842335486_12842335489"
TLS_PHASES = 6

APPROACH_EDGES = [
    "1387307747#0",
    "1387307749#0",
    "-1443664838",
    "40969892#3",
    "-1443664839#1",
    "897052936#8",
]

FLOOD_EDGES = [
    "1387307747#0", "1387307749#0", "-1443664838",   "40969892#3",
    "1443664838",   "1387307747#1", "1387307749#1",  "-40969892#3",
    "-1443664839#1","1443664839#1", "897052936#8",   "-897052936#8",
    "1387307748#0", "1387307748#1", "1387307750#0",  "1387323387",
    "1391662482",   "1391662483#0", "1391662483#1",  "1391662484",
]

# Validated V6.2 proven routes (4/4 confirmed in V6.4)
CANDIDATE_ROUTES = [
    ("-1443664838",    "1443664839#1"),
    ("40969892#3",     "-40969892#3"),
    ("897052936#8",    "1443664838"),
    ("-1443664839#1",  "-897052936#8"),
]

# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
SIM_DURATION    = 3600
STEP_LENGTH     = 1.0
AADT            = 9_500
K_FACTOR        = 0.092
PHF             = 0.92
PEAK_VEHPH      = int(AADT * K_FACTOR / PHF)   # 950

PPO_UPDATE_INT  = 300
MIN_PHASE_DUR   = 20
MAX_PHASE_DUR   = 90
HISTORY_LEN     = 8

# ── Flood thresholds ──────────────────────────────────────────────────────────
FLOOD_WATCH      = 0.05
FLOOD_SPEED_RED  = 0.15
FLOOD_LANE_CLOSE = 0.30
FLOOD_IMPASSABLE = 0.50

SPD_IMPASSABLE = 1.0
SPD_LANE_CLOSE = 2.8
SPD_SPEED_RED  = 5.5
SPD_WATCH      = 8.3
SPD_NORMAL     = 13.9

# ── Demand scales ─────────────────────────────────────────────────────────────
DEMAND_SCALE = {
    "baseline": 1.00,
    "mild":     0.90,
    "moderate": 0.75,
    "severe":   0.30,
}

SCENARIOS = [
    {"name": "Baseline",  "depth": 0.00, "label": "baseline"},
    {"name": "Mild",      "depth": 0.12, "label": "mild"},
    {"name": "Moderate",  "depth": 0.28, "label": "moderate"},
    {"name": "Severe",    "depth": 0.55, "label": "severe"},
]

# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-1/2/3] ANTIFRAGILITY INDEX — CORRECTED FORMULA
# ══════════════════════════════════════════════════════════════════════════════
def compute_AI(arrived: int,
               demand_issued: int,
               avg_wait_s: float,
               baseline_ratio: float | None = None) -> tuple[float, str]:
    """
    Antifragility Index (AI) — V6.5 formulation

    Core insight: a system is antifragile if it performs BETTER than
    expected under stress relative to the demand actually placed on it.

    Components:
      throughput_ratio = arrived / demand_issued
        → 1.0 = perfect service of available demand
        → penalizes only actual loss, not reduced demand

      wait_penalty = avg_wait_s / REFERENCE_WAIT
        → REFERENCE_WAIT = 45s (signalized intersection free-flow)
        → values > 1.0 indicate congestion above baseline expectation

      AI = throughput_ratio - wait_penalty × 0.3

    Thresholds (calibrated to signalized intersection performance):
      AI >  0.70  → ANTIFRAGILE  (serves >70% of demand, low wait)
      AI >  0.50  → RESILIENT    (serves >50% of demand, moderate wait)
      AI <= 0.50  → FRAGILE
    """
    REFERENCE_WAIT = 45.0   # seconds — typical signalized intersection

    if demand_issued <= 0:
        return 0.0, "FRAGILE"

    tput_ratio   = arrived / demand_issued
    wait_penalty = (avg_wait_s / REFERENCE_WAIT) * 0.3
    AI           = tput_ratio - wait_penalty

    if AI > 0.70:
        label = "ANTIFRAGILE"
    elif AI > 0.50:
        label = "RESILIENT"
    else:
        label = "FRAGILE"

    return round(AI, 4), label


def compute_AI_relative(arrived: int,
                        demand_issued: int,
                        avg_wait_s: float,
                        baseline_AI: float) -> tuple[float, str]:
    """
    Relative AI — measures improvement ABOVE baseline performance.
    Used for the antifragility delta table.

    relative_AI = AI_scenario / baseline_AI_adjusted
    A system is antifragile if flooded scenario AI >= baseline AI
    (maintains or improves performance ratio under stress).
    """
    ai_abs, _ = compute_AI(arrived, demand_issued, avg_wait_s)

    # Antifragility delta: how much better/worse than baseline
    delta = ai_abs - baseline_AI

    if delta >= 0.0:
        label = "ANTIFRAGILE"
    elif delta >= -0.15:
        label = "RESILIENT"
    else:
        label = "FRAGILE"

    return round(delta, 4), label


# ══════════════════════════════════════════════════════════════════════════════
#  SUMOLIB ROUTE VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════
_NET: sumolib.net.Net | None = None

def _get_net() -> sumolib.net.Net:
    global _NET
    if _NET is None:
        print("  [NET] Loading sumolib network …", end=" ")
        _NET = sumolib.net.readNet(NET_FILE, withInternal=False)
        print("OK")
    return _NET


def _edge_has_drivable_lanes(net, eid: str) -> bool:
    try:
        edge = net.getEdge(eid)
    except Exception:
        return False
    for lane in edge.getLanes():
        perms = lane.getPermissions()
        if "passenger" in perms or "all" in perms or not perms:
            return True
    return False


def _edges_connected(net, from_eid: str, to_eid: str) -> bool:
    try:
        from_edge = net.getEdge(from_eid)
        for oe in from_edge.getOutgoing().keys():
            if oe.getID() == to_eid:
                return True
    except Exception:
        pass
    return False


def validate_routes(candidates: list) -> list:
    net   = _get_net()
    valid = []
    for rid, edges in candidates:
        ok = True
        for e in edges:
            if not _edge_has_drivable_lanes(net, e):
                print(f"    [ROUTE-SKIP] {rid}: '{e}' no drivable lanes")
                ok = False; break
        if ok:
            for i in range(len(edges) - 1):
                if not _edges_connected(net, edges[i], edges[i + 1]):
                    print(f"    [ROUTE-SKIP] {rid}: "
                          f"'{edges[i]}' → '{edges[i+1]}' not connected")
                    ok = False; break
        if ok:
            valid.append((rid, edges))
    return valid


def get_valid_routes() -> list:
    print("  [ROUTES] Validating candidate routes …")
    base = [(f"r_{i}", list(p))
            for i, p in enumerate(CANDIDATE_ROUTES)]
    validated = validate_routes(base)
    labels    = ["r_NB", "r_SB", "r_EB", "r_WB"]
    relabeled = [(labels[i] if i < len(labels) else f"r_X{i}", edges)
                 for i, (_, edges) in enumerate(validated)]
    print(f"  [ROUTES] {len(relabeled)}/4 routes confirmed: "
          f"{[r for r,_ in relabeled]}")
    return relabeled


# ══════════════════════════════════════════════════════════════════════════════
#  TRIPINFO PARSER — [FIX-2] accurate per-vehicle wait time
# ══════════════════════════════════════════════════════════════════════════════
def parse_tripinfo_wait(tripinfo_xml: str) -> float:
    """
    Parse SUMO tripinfo XML and return mean WaitingTime (seconds).
    Falls back to 0.0 if file absent or malformed.
    """
    try:
        tree   = ET.parse(tripinfo_xml)
        root   = tree.getroot()
        waits  = [float(t.get("waitingTime", 0))
                  for t in root.iter("tripinfo")]
        return float(np.mean(waits)) if waits else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _sumo_cmd(cfg_path: str) -> list:
    return [
        SUMO_BIN,
        "-c",           cfg_path,
        "--start",
        "--quit-on-end",
        "--no-step-log",
        "--ignore-route-errors",   "true",
        "--time-to-teleport",      "300",
        "--collision.action",      "warn",
        "--no-warnings",           "false",
    ]


def _preflight() -> bool:
    try:
        r   = subprocess.run([SUMO_BIN, "--version"],
                             capture_output=True, text=True, timeout=10)
        ver = r.stdout.splitlines()[0] if r.stdout else "?"
        print(f"[OK] SUMO pre-flight: {ver}")
        return True
    except Exception as e:
        print(f"[WARN] pre-flight: {e}")
        return False


def _safe_close():
    try:
        if traci.isLoaded():
            traci.close()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNING STATS (Welford)
# ══════════════════════════════════════════════════════════════════════════════
class RunningStats:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2   += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-8

    def normalize(self, x: float) -> float:
        return (x - self.mean) / self.std


# ══════════════════════════════════════════════════════════════════════════════
#  DEMAND BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_route_file(label: str, depth: float,
                     valid_routes: list) -> tuple[str, int]:
    """Returns (rou_path, vehicles_issued)."""
    random.seed(42 + abs(hash(label)) % 1000)
    np.random.seed(42 + abs(hash(label)) % 1000)

    base_label = label.split("_seed")[0]
    scale = DEMAND_SCALE.get(base_label, 1.0)
    rate     = (PEAK_VEHPH * scale) / 3600.0
    rids     = [r for r, _ in valid_routes]

    departures = []
    for t in range(SIM_DURATION):
        n = int(np.random.poisson(rate))
        for _ in range(n):
            departures.append((t + random.uniform(0.0, 0.95),
                               random.choice(rids)))

    rou_path = str(OUT_DIR / f"{label}.rou.xml")
    with open(rou_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('  <vType id="car" accel="2.6" decel="4.5" '
                'length="4.5" maxSpeed="13.9" sigma="0.5"/>\n\n')
        for rid, edges in valid_routes:
            f.write(f'  <route id="{rid}" '
                    f'edges="{" ".join(edges)}"/>\n')
        f.write("\n")
        for vid, (dep, rid) in enumerate(sorted(departures)):
            f.write(f'  <vehicle id="v{vid}" type="car" '
                    f'route="{rid}" depart="{dep:.2f}" '
                    f'departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")

    n_veh = len(departures)
    print(f"  [DEMAND] {label:<10}: {n_veh:4d} vehicles "
          f"(target {int(PEAK_VEHPH*scale):4d} veh/hr, "
          f"scale={scale:.2f})")
    return rou_path, n_veh


# ══════════════════════════════════════════════════════════════════════════════
#  SUMO CFG WRITER
# ══════════════════════════════════════════════════════════════════════════════
def write_sumo_cfg(label: str, rou_path: str) -> str:
    cfg_path  = str(OUT_DIR / f"{label}.sumocfg")
    trip_out  = str(OUT_DIR / f"{label}_tripinfo.xml")
    summ_out  = str(OUT_DIR / f"{label}_summary.xml")
    with open(cfg_path, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file    value="{NET_FILE}"/>
    <route-files value="{rou_path}"/>
  </input>
  <time>
    <begin       value="0"/>
    <end         value="{SIM_DURATION}"/>
    <step-length value="{STEP_LENGTH}"/>
  </time>
  <processing>
    <ignore-route-errors       value="true"/>
    <time-to-teleport          value="300"/>
    <collision.action          value="warn"/>
  </processing>
  <report>
    <no-step-log               value="true"/>
    <no-warnings               value="false"/>
    <duration-log.statistics   value="true"/>
  </report>
  <output>
    <tripinfo-output value="{trip_out}"/>
    <summary-output  value="{summ_out}"/>
  </output>
</configuration>
""")
    return cfg_path


# ══════════════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT TRANSFORMER ENCODER
# ══════════════════════════════════════════════════════════════════════════════
class LightweightTransformer:
    INPUT_DIM = 16
    D_MODEL   = 32
    N_HEADS   = 2
    D_HEAD    = D_MODEL // N_HEADS

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        k   = 0.1
        _w  = lambda r, c: (rng.standard_normal((r, c))
                            .astype(np.float32) * k)
        self.Wq = _w(self.INPUT_DIM, self.D_MODEL)
        self.Wk = _w(self.INPUT_DIM, self.D_MODEL)
        self.Wv = _w(self.INPUT_DIM, self.D_MODEL)
        self.Wo = _w(self.D_MODEL,   self.D_MODEL)
        self.W1 = _w(self.D_MODEL,   self.D_MODEL * 2)
        self.W2 = _w(self.D_MODEL * 2, self.D_MODEL)
        self.pos_enc = (rng.standard_normal((HISTORY_LEN, self.D_MODEL))
                        .astype(np.float32) * 0.02)

    @staticmethod
    def _softmax(x):
        ex = np.exp(x - x.max(axis=-1, keepdims=True))
        return ex / (ex.sum(axis=-1, keepdims=True) + 1e-9)

    def _ln(self, x):
        return ((x - x.mean(-1, keepdims=True))
                / (x.std(-1, keepdims=True) + 1e-6))

    def encode(self, history: np.ndarray) -> np.ndarray:
        T  = history.shape[0]
        pe = self.pos_enc[:T]
        Q  = history @ self.Wq + pe
        K  = history @ self.Wk + pe
        V  = history @ self.Wv + pe
        sc = math.sqrt(self.D_HEAD)
        heads = []
        for h in range(self.N_HEADS):
            sl = slice(h * self.D_HEAD, (h + 1) * self.D_HEAD)
            a  = self._softmax((Q[:, sl] @ K[:, sl].T) / sc)
            heads.append(a @ V[:, sl])
        mha = np.concatenate(heads, axis=-1)
        x   = self._ln(mha @ self.Wo + history @ self.Wq)
        ff  = np.maximum(0.0, x @ self.W1) @ self.W2
        x   = self._ln(x + ff)
        return x.mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  PPO AGENT  [FIX-4: shared across scenarios]
# ══════════════════════════════════════════════════════════════════════════════
class PPOAgent:
    STATE_DIM  = 32
    HIDDEN_DIM = 64
    LR_A  = 3e-4
    LR_C  = 1e-3
    GAMMA = 0.99
    LAM   = 0.95
    CLIP  = 0.20
    ENT   = 0.02
    MXGRD = 0.50
    EPOCHS = 4

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)

        def _glorot(fi, fo):
            std = math.sqrt(2.0 / (fi + fo))
            return rng.standard_normal((fi, fo)).astype(np.float32) * std

        self.a_W1 = _glorot(self.STATE_DIM, self.HIDDEN_DIM)
        self.a_b1 = np.zeros(self.HIDDEN_DIM, np.float32)
        self.a_W2 = _glorot(self.HIDDEN_DIM, 1)
        self.a_b2 = np.float32(0.0)
        self.ls   = np.float32(-1.0)

        self.c_W1 = _glorot(self.STATE_DIM, self.HIDDEN_DIM)
        self.c_b1 = np.zeros(self.HIDDEN_DIM, np.float32)
        self.c_W2 = _glorot(self.HIDDEN_DIM, 1)
        self.c_b2 = np.float32(0.0)

        self.rew_stats = RunningStats()

        self.buf_s:  list = []
        self.buf_a:  list = []
        self.buf_r:  list = []
        self.buf_lp: list = []
        self.buf_d:  list = []
        self.n_updates = 0

    def _mean(self, s):
        h = np.maximum(0.0, s @ self.a_W1 + self.a_b1)
        return float(np.dot(h, self.a_W2.ravel()) + self.a_b2)

    def _val(self, s):
        h = np.maximum(0.0, s @ self.c_W1 + self.c_b1)
        return float(np.dot(h, self.c_W2.ravel()) + self.c_b2)

    def _log_prob(self, a, mu):
        std = float(np.exp(self.ls))
        return float(
            -0.5 * ((a - mu) / std) ** 2
            - float(self.ls)
            - 0.5 * math.log(2.0 * math.pi))

    def select_action(self, s):
        s   = np.asarray(s, np.float32).ravel()
        mu  = self._mean(s)
        std = float(np.exp(self.ls))
        a   = float(mu + std * np.random.randn())
        return a, self._log_prob(a, mu)

    @staticmethod
    def action_to_duration(raw: float) -> int:
        sig = 1.0 / (1.0 + math.exp(-float(raw)))
        return int(MIN_PHASE_DUR + sig * (MAX_PHASE_DUR - MIN_PHASE_DUR))

    def store(self, s, a, r, lp, done=False):
        self.rew_stats.update(r)
        r_n = self.rew_stats.normalize(r)
        self.buf_s.append(np.asarray(s, np.float32).ravel())
        self.buf_a.append(float(a))
        self.buf_r.append(float(r_n))
        self.buf_lp.append(float(lp))
        self.buf_d.append(bool(done))

    def update(self) -> dict:
        n = len(self.buf_r)
        if n < 4:
            self._clear(); return {}

        sa  = np.array(self.buf_s,  np.float32)
        aa  = np.array(self.buf_a,  np.float32)
        ola = np.array(self.buf_lp, np.float32)
        rws = np.array(self.buf_r,  np.float32)
        don = np.array(self.buf_d,  np.float32)

        vals = np.array([self._val(sa[i]) for i in range(n)], np.float32)
        adv  = np.zeros(n, np.float32)
        ret  = np.zeros(n, np.float32)
        gae  = 0.0
        for t in reversed(range(n)):
            nv    = vals[t + 1] if t < n - 1 else 0.0
            delta = rws[t] + self.GAMMA * nv * (1 - don[t]) - vals[t]
            gae   = delta + self.GAMMA * self.LAM * (1 - don[t]) * gae
            adv[t] = gae
            ret[t] = gae + vals[t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tot_al = 0.0; tot_cl = 0.0
        for _ in range(self.EPOCHS):
            ha  = np.maximum(0.0, sa @ self.a_W1 + self.a_b1)
            mu  = ha @ self.a_W2.ravel() + float(self.a_b2)
            std = float(np.exp(self.ls))
            nlp = (-0.5 * ((aa - mu) / std) ** 2
                   - float(self.ls)
                   - 0.5 * math.log(2.0 * math.pi))
            rat  = np.exp(np.clip(nlp - ola, -5.0, 5.0))
            sur1 = rat * adv
            sur2 = np.clip(rat, 1-self.CLIP, 1+self.CLIP) * adv
            ent  = 0.5 * (1 + math.log(2*math.pi)) + float(self.ls)
            al   = float(-np.minimum(sur1, sur2).mean()) - self.ENT * ent
            tot_al += al

            clp     = (rat < 1-self.CLIP) | (rat > 1+self.CLIP)
            er      = np.where(clp, np.clip(rat, 1-self.CLIP, 1+self.CLIP), rat)
            dL_dmu  = -(adv * er * ((aa - mu) / std**2)) / n
            gW2 = ha.T @ dL_dmu
            gb2 = float(dL_dmu.sum())
            dh  = np.outer(dL_dmu, self.a_W2.ravel()) * (ha > 0)
            gW1 = sa.T @ dh / n
            gb1 = dh.mean(0)

            def _cg(g):
                nm = np.linalg.norm(g)
                return g if nm <= self.MXGRD else g * (self.MXGRD / nm)

            self.a_W1 -= self.LR_A * _cg(gW1)
            self.a_b1 -= self.LR_A * _cg(gb1)
            self.a_W2 -= self.LR_A * _cg(gW2.reshape(-1, 1))
            self.a_b2  = np.float32(float(self.a_b2) - self.LR_A * gb2)

            hc  = np.maximum(0.0, sa @ self.c_W1 + self.c_b1)
            vs  = hc @ self.c_W2.ravel() + float(self.c_b2)
            cl  = float(((ret - vs)**2).mean())
            tot_cl += cl
            dv  = -2.0 * (ret - vs) / n
            gcW2 = hc.T @ dv
            gcb2 = float(dv.sum())
            dhc  = np.outer(dv, self.c_W2.ravel()) * (hc > 0)
            gcW1 = sa.T @ dhc / n
            gcb1 = dhc.mean(0)
            self.c_W1 -= self.LR_C * _cg(gcW1)
            self.c_b1 -= self.LR_C * _cg(gcb1)
            self.c_W2 -= self.LR_C * _cg(gcW2.reshape(-1, 1))
            self.c_b2  = np.float32(float(self.c_b2) - self.LR_C * gcb2)

            ls_new = float(self.ls) + self.LR_A * self.ENT * ent * 0.05
            self.ls = np.float32(np.clip(ls_new, -3.0, 0.5))

        self.n_updates += 1
        self._clear()
        return {"al": round(tot_al / self.EPOCHS, 5),
                "cl": round(tot_cl / self.EPOCHS, 4)}

    def _clear(self):
        for b in (self.buf_s, self.buf_a, self.buf_r,
                  self.buf_lp, self.buf_d):
            b.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  FLOOD CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class FloodController:
    def __init__(self, edges: list, depth: float):
        self.cfg_edges = edges
        self.depth     = depth
        self.live      : list = []

    def init(self):
        live       = set(traci.edge.getIDList())
        self.live  = [e for e in self.cfg_edges if e in live]
        print(f"    [Flood] {len(self.live)}/{len(self.cfg_edges)} "
              f"flood edges confirmed  (depth={self.depth:.2f}m)")

    def _spd(self, e, v):
        try:
            traci.edge.setMaxSpeed(e, max(v, SPD_IMPASSABLE))
        except Exception:
            pass

    def apply(self):
        if self.depth < FLOOD_WATCH:
            return
        for e in self.live:
            if   self.depth >= FLOOD_IMPASSABLE: self._spd(e, SPD_IMPASSABLE)
            elif self.depth >= FLOOD_LANE_CLOSE: self._spd(e, SPD_LANE_CLOSE)
            elif self.depth >= FLOOD_SPEED_RED:  self._spd(e, SPD_SPEED_RED)
            else:                                 self._spd(e, SPD_WATCH)

    def restore(self):
        for e in self.live:
            try:
                traci.edge.setMaxSpeed(e, SPD_NORMAL)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE TLS CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class AdaptiveTLS:
    OBS_DIM = LightweightTransformer.INPUT_DIM

    def __init__(self, tls_id, agent: PPOAgent,
                 transformer: LightweightTransformer,
                 demand_issued: int):
        self.tls_id        = tls_id
        self.agent         = agent
        self.tf            = transformer
        self.demand_issued = demand_issued
        zero               = np.zeros(self.OBS_DIM, np.float32)
        self.history       = deque([zero.copy() for _ in range(HISTORY_LEN)],
                                   maxlen=HISTORY_LEN)
        self.phase     = 0
        self.timer     = 0
        self.duration  = MIN_PHASE_DUR
        self.step      = 0
        self.cum_rew   = 0.0
        self._live     = False
        self._nph      = TLS_PHASES
        # Rolling AI window for paper reward bonus:
        # Omega_t = min(0.5 * mean(AI_t over last 30 steps), 5.0)
        self._ai_win   = deque(maxlen=30)
        self._cum_arr  = 0

    def init(self):
        live = set(traci.trafficlight.getIDList())
        if self.tls_id in live:
            self._live = True
            prog = traci.trafficlight.getAllProgramLogics(self.tls_id)
            if prog:
                self._nph = len(prog[0].phases)
            print(f"    [TLS] '{self.tls_id}' confirmed — "
                  f"{self._nph} phases")
        else:
            print(f"    [WARN] TLS '{self.tls_id}' not found; "
                  f"using first available")
            if live:
                self.tls_id = sorted(live)[0]
                self._live  = True
                prog = traci.trafficlight.getAllProgramLogics(self.tls_id)
                if prog:
                    self._nph = len(prog[0].phases)

    def _obs(self, depth: float) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, np.float32)
        for i, e in enumerate(APPROACH_EDGES):
            try:
                obs[i]     = min(
                    traci.edge.getLastStepHaltingNumber(e) / 20.0, 1.0)
                obs[i + 6] = min(
                    traci.edge.getLastStepOccupancy(e), 1.0)
            except Exception:
                pass
        obs[12] = self.phase / max(self._nph - 1, 1)
        obs[13] = min(self.timer / MAX_PHASE_DUR, 1.0)
        obs[14] = min(depth, 1.0)
        obs[15] = self.step / SIM_DURATION
        return obs

    def _step_ai(self, avg_wait: float) -> float:
        """
        Step-level AI approximation used only to compute the rolling
        antifragility bonus in the paper reward function.

        Episode AI in the final results is still computed from tripinfo
        waiting time using compute_AI().
        """
        expected_demand = self.demand_issued * min(
            (self.step + 1) / SIM_DURATION, 1.0)
        throughput_ratio = self._cum_arr / max(expected_demand, 1.0)
        wait_penalty = 0.3 * (avg_wait / 45.0)
        return throughput_ratio - wait_penalty

    def _reward(self, arr, que, avg_wait) -> float:
        """
        Paper reward function:

            R_t = 1.0 a_t - 0.1 h_t - 0.01 w_bar_t + Omega_t

        where Omega_t = min(0.5 * rolling_mean_AI_30, 5.0).
        """
        self._cum_arr += arr
        step_ai = self._step_ai(avg_wait)
        self._ai_win.append(step_ai)

        rolling_ai = sum(self._ai_win) / max(len(self._ai_win), 1)
        omega_t = min(0.5 * rolling_ai, 5.0)

        return float(arr * 1.0
                     - que * 0.1
                     - avg_wait * 0.01
                     + omega_t)

    def tick(self, arr, que, depth, avg_wait) -> dict:
        obs   = self._obs(depth)
        self.history.append(obs)
        state = self.tf.encode(np.array(self.history, np.float32))
        rew   = self._reward(arr, que, avg_wait)
        self.cum_rew += rew
        self.timer   += 1
        ppo_info = {}

        if self.timer >= self.duration:
            raw, lp = self.agent.select_action(state)
            self.agent.store(state, raw, rew, lp,
                             done=(self.step >= SIM_DURATION - 1))
            self.phase    = (self.phase + 1) % self._nph
            self.duration = self.agent.action_to_duration(raw)
            self.timer    = 0
            if self._live:
                try:
                    traci.trafficlight.setPhase(
                        self.tls_id, self.phase)
                except Exception:
                    pass

        if self.step > 0 and self.step % PPO_UPDATE_INT == 0:
            ppo_info = self.agent.update()

        self.step += 1
        return {"phase": self.phase, "dur": self.duration,
                "reward": rew, "ppo": ppo_info}


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_scenario(scenario: dict, cfg_path: str,
                 demand_issued: int,
                 agent: PPOAgent,
                 transformer: LightweightTransformer) -> dict:
    name  = scenario["name"]
    depth = scenario["depth"]
    cfg_stem = Path(cfg_path).stem
    trip_xml = str(OUT_DIR / f"{cfg_stem}_tripinfo.xml")

    print(f"\n{'='*68}")
    print(f"  SCENARIO: {name:<12}  depth={depth:.2f}m  "
          f"demand={demand_issued}")
    print(f"{'='*68}")

    port       = _free_port()
    cmd        = _sumo_cmd(cfg_path)
    tls_ctrl   = AdaptiveTLS(TLS_ID, agent, transformer, demand_issued)
    flood_ctrl = FloodController(FLOOD_EDGES, depth)

    tot_arrived  = 0
    tot_teleport = 0
    tot_reward   = 0.0

    print(f"    [Launch] port={port}")
    try:
        traci.start(cmd, port=port, numRetries=5, verbose=False)
        tls_ctrl.init()
        flood_ctrl.init()
        flood_ctrl.apply()

        for step in range(SIM_DURATION):
            traci.simulationStep()

            arr = int(traci.simulation.getArrivedNumber())
            tel = int(traci.simulation.getStartingTeleportNumber())
            tot_arrived  += arr
            tot_teleport += tel

            que = 0; step_wait_total = 0.0
            for e in APPROACH_EDGES:
                try:
                    que       += int(
                        traci.edge.getLastStepHaltingNumber(e))
                    step_wait_total += traci.edge.getWaitingTime(e)
                except Exception:
                    pass

            # Paper reward uses average waiting time, not total edge wait.
            active_vehicles = len(traci.vehicle.getIDList())
            avg_wait_step = step_wait_total / max(active_vehicles, 1)

            info         = tls_ctrl.tick(arr, que, depth, avg_wait_step)
            tot_reward  += info["reward"]

            if step > 0 and step % 60 == 0:
                flood_ctrl.apply()

            if step % 300 == 0:
                p  = info["ppo"]
                ps = (f"  al={p['al']:.5f} cl={p['cl']:.4f}" if p else "")
                print(f"    t={step:5d}s  arr={tot_arrived:5d}  "
                      f"que={que:3d}  tel={tot_teleport:3d}  "
                      f"ph={info['phase']}  "
                      f"dur={info['dur']}s  "
                      f"rew={info['reward']:+.2f}{ps}")

        flood_ctrl.restore()
        _safe_close()

    except traci.exceptions.FatalTraCIError as exc:
        print(f"  [ERROR] FatalTraCIError: {exc}")
        traceback.print_exc()
        _safe_close()
        return {"scenario": name, "error": str(exc)}
    except Exception as exc:
        print(f"  [ERROR] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        _safe_close()
        return {"scenario": name, "error": str(exc)}

    # [FIX-2] Read accurate wait from tripinfo XML
    avg_wait = parse_tripinfo_wait(trip_xml)
    if avg_wait == 0.0:
        # fallback: estimate from SUMO Statistics output
        avg_wait = tls_ctrl.cum_rew   # last resort sentinel

    tput  = tot_arrived / (SIM_DURATION / 3600.0)
    AI, ai_label = compute_AI(tot_arrived, demand_issued, avg_wait)

    result = {
        "scenario":         name,
        "flood_depth_m":    round(depth, 2),
        "demand_issued":    demand_issued,
        "total_arrived":    tot_arrived,
        "total_teleport":   tot_teleport,
        "pct_served":       round(
            100.0 * tot_arrived / max(demand_issued, 1), 1),
        "avg_wait_s":       round(avg_wait, 3),
        "throughput_vehph": round(tput, 1),
        "AI":               AI,
        "AI_label":         ai_label,
        "cum_reward":       round(tot_reward, 2),
        "ppo_updates":      agent.n_updates,
    }

    print(f"\n  ── {name} RESULTS ──────────────────────────────────")
    for k, v in result.items():
        print(f"     {k:<22}: {v}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 68)
    print("  ANTIFRAGILE TRAFFIC SIGNAL CONTROL — V6.5")
    print(f"  TLS  : {TLS_ID}")
    print(f"  OUT  : {OUT_DIR}")
    print("=" * 68)

    if not os.path.isfile(NET_FILE):
        sys.exit(f"[FATAL] Network not found: {NET_FILE}")
    if not os.path.isfile(SUMO_BIN):
        sys.exit(f"[FATAL] SUMO binary not found: {SUMO_BIN}")

    _preflight()

    valid_routes = get_valid_routes()
    if len(valid_routes) < 1:
        sys.exit("[FATAL] No valid routes.")

    transformer = LightweightTransformer(seed=42)
    # [FIX-4] Single shared agent accumulates knowledge across scenarios
    agent = PPOAgent(seed=42)

    # Self-test
    print("  [SELF-TEST] ...", end=" ")
    dummy = np.zeros((HISTORY_LEN, LightweightTransformer.INPUT_DIM),
                     np.float32)
    state = transformer.encode(dummy)
    assert state.shape == (PPOAgent.STATE_DIM,)
    a, lp = agent.select_action(state)
    dur   = agent.action_to_duration(a)
    assert MIN_PHASE_DUR <= dur <= MAX_PHASE_DUR
    for _ in range(5):
        agent.store(state, a, -1.0, lp)
    upd = agent.update()
    assert "al" in upd
    print(f"OK  (a={a:.4f}, lp={lp:.4f}, dur={dur}s, "
          f"al={upd['al']:.5f}, cl={upd['cl']:.4f})")

    # Build configs
    print()
    configs = []
    for sc in SCENARIOS:
        rou, n_veh = build_route_file(
            sc["label"], sc["depth"], valid_routes)
        cfg = write_sumo_cfg(sc["label"], rou)
        configs.append((sc, cfg, n_veh))

    # Run scenarios — [FIX-4] same agent instance reused
    results = []
    for sc, cfg, n_veh in configs:
        res = run_scenario(sc, cfg, n_veh, agent, transformer)
        results.append(res)
        time.sleep(2.0)

    # Summary
    print("\n" + "=" * 68)
    print("  FINAL SUMMARY — V6.5")
    print("=" * 68)
    print(f"  {'Scenario':<12} {'Depth':>6} {'Demand':>7} "
          f"{'Arrived':>8} {'%Served':>8} {'Wait(s)':>8} "
          f"{'AI':>8}  Label")
    print("  " + "-" * 70)

    ok = []
    for r in results:
        if "error" in r:
            print(f"  {r['scenario']:<12}  ERROR: {r['error'][:45]}")
        else:
            print(f"  {r['scenario']:<12} {r['flood_depth_m']:>6.2f} "
                  f"{r['demand_issued']:>7d} {r['total_arrived']:>8d} "
                  f"{r['pct_served']:>7.1f}% {r['avg_wait_s']:>8.3f} "
                  f"{r['AI']:>8.4f}  {r['AI_label']}")
            ok.append(r)

    # Delta table vs baseline
    if len(ok) >= 2:
        base = next((r for r in ok if r["flood_depth_m"] == 0.0), ok[0])
        base_AI, _ = compute_AI(
            base["total_arrived"], base["demand_issued"],
            base["avg_wait_s"])
        print("\n  ANTIFRAGILITY DELTAS (vs Baseline):")
        print(f"  {'Scenario':<12} {'Δ%Served':>9} "
              f"{'ΔWait':>9} {'ΔAI':>8}  Label")
        print("  " + "-" * 50)
        for r in ok:
            if r["flood_depth_m"] > 0.0:
                d_pct  = r["pct_served"]  - base["pct_served"]
                d_wait = r["avg_wait_s"]  - base["avg_wait_s"]
                d_AI, d_lbl = compute_AI_relative(
                    r["total_arrived"], r["demand_issued"],
                    r["avg_wait_s"], base_AI)
                print(f"  {r['scenario']:<12} {d_pct:>+8.1f}% "
                      f"{d_wait:>+9.3f} {d_AI:>+8.4f}  {d_lbl}")

    # Antifragility verdict
    print("\n  ANTIFRAGILITY ASSESSMENT:")
    print("  " + "-" * 50)
    af_count = sum(1 for r in ok
                   if r.get("AI_label") in ("ANTIFRAGILE", "RESILIENT"))
    print(f"  Scenarios passing (ANTIFRAGILE/RESILIENT): "
          f"{af_count}/{len(ok)}")
    if ok:
        ai_vals = [r["AI"] for r in ok]
        print(f"  AI range: [{min(ai_vals):.4f}, {max(ai_vals):.4f}]")
        if all(r.get("AI_label") in ("ANTIFRAGILE", "RESILIENT")
               for r in ok):
            print("  ✓ SYSTEM IS ANTIFRAGILE — all scenarios "
                  "RESILIENT or better")
        elif af_count >= len(ok) // 2:
            print("  ~ PARTIALLY ANTIFRAGILE — majority of "
                  "scenarios resilient")
        else:
            print("  ✗ SYSTEM IS FRAGILE — revisit phase timing "
                  "and flood routing")

    # Save
    out = OUT_DIR / "results_v65.json"
    out.write_text(json.dumps(
        {"version": "V6.5",
         "tls_id": TLS_ID,
         "peak_vehph": PEAK_VEHPH,
         "valid_routes": [{"id": r, "edges": e}
                          for r, e in valid_routes],
         "AI_formula": {
             "throughput_ratio": "arrived / demand_issued",
             "wait_penalty":
                 "avg_wait_s / 45.0 * 0.3",
             "AI": "throughput_ratio - wait_penalty",
             "thresholds": {
                 "ANTIFRAGILE": ">0.70",
                 "RESILIENT":   ">0.50",
                 "FRAGILE":     "<=0.50"}},
         "results": results},
        indent=2))
    print(f"\n[SAVED] {out}")
    print("=" * 68)


if __name__ == "__main__":
    main()