# corrected_constants.py
# Auto-generated from edge_discovery.py results
# Buist Ave & Spruill Ave, North Charleston SC (FEMA Zone AE)

INTERSECTION_LAT = 32.8733
INTERSECTION_LON = -79.9801

TLS_ID      = "cluster_110099785_12842335486_12842335489"
JUNCTION_ID = "cluster_110099785_12842335486_12842335489"
FEMA_ZONE   = "AE"

# Confirmed approach edges (vehicles arriving AT intersection)
APPROACH_EDGES = [
    "1387307747#0",   # dist=7m,   lanes=1
    "1387307749#0",   # dist=7m,   lanes=1
    "-1443664838",    # dist=7m,   lanes=1
    "40969892#3",     # dist=15m,  lanes=1
    "-1443664839#1",  # dist=26m,  lanes=2
    "897052936#8",    # dist=100m, lanes=1
]

# Confirmed exit edges (vehicles departing intersection)
EXIT_EDGES = [
    "1443664838",     # dist=8m,   lanes=1
    "1387307747#1",   # dist=12m,  lanes=1
    "1387307749#1",   # dist=12m,  lanes=1
    "-40969892#3",    # dist=15m,  lanes=1
    "1443664839#1",   # dist=26m,  lanes=1
    "-897052936#8",   # dist=100m, lanes=1
]

# All edges subject to flood speed reduction (approach + exit)
FLOOD_EDGES = APPROACH_EDGES + EXIT_EDGES

# Confirmed valid 2-edge routes through intersection
VALID_ROUTES = [
    ["-1443664838",   "1443664838"],
    ["-1443664838",   "-40969892#3"],
    ["-1443664838",   "1443664839#1"],
    ["-1443664838",   "-897052936#8"],
    ["40969892#3",    "1443664838"],
    ["40969892#3",    "-40969892#3"],
    ["40969892#3",    "1443664839#1"],
    ["40969892#3",    "-897052936#8"],
    ["-1443664839#1", "1443664838"],
    ["-1443664839#1", "-40969892#3"],
    ["-1443664839#1", "1443664839#1"],
    ["-1443664839#1", "-897052936#8"],
    ["897052936#8",   "1443664838"],
    ["897052936#8",   "-40969892#3"],
    ["897052936#8",   "1443664839#1"],
    ["897052936#8",   "-897052936#8"],
]

# Flood depth scenarios (metres)
FLOOD_SCENARIOS = {
    "Baseline": 0.00,
    "Minor":    0.15,
    "Moderate": 0.30,
    "Major":    0.45,
    "Severe":   0.55,
    "Extreme":  0.80,
}

# Speed reduction factor per flood depth (m/s floor = 1.0)
def flood_speed_factor(depth_m):
    if depth_m <= 0.0:
        return 1.00
    elif depth_m <= 0.15:
        return 0.80
    elif depth_m <= 0.30:
        return 0.60
    elif depth_m <= 0.45:
        return 0.40
    elif depth_m <= 0.55:
        return 0.25
    else:
        return 0.10
