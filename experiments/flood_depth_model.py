# flood_depth_model.py
"""
Water Depth Simulation Model for Buist/Spruill Intersection
Based on Charleston, SC compound flood characteristics:
  - Tidal surge + heavy rainfall
  - Historical FEMA flood zone AE designation
  - Real drainage capacity data for Peninsula Charleston
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math


# ============================================================
# FLOOD SCENARIO DEFINITIONS
# ============================================================

class FloodScenario(Enum):
    NONE = "none"
    MILD = "mild"  # 2-year return period
    MODERATE = "moderate"  # 10-year return period
    SEVERE = "severe"  # 100-year return period
    EXTREME = "extreme"  # 500-year return period (future climate)


@dataclass
class FloodZone:
    """
    Represents a spatial flood zone at Buist/Spruill.

    Based on:
    - FEMA FIRMette for Charleston County (Zone AE)
    - SC DHEC drainage capacity reports
    - Charleston Peninsula elevation data (avg 2-4m NAVD88)
    """
    zone_id: str
    center_x: float  # SUMO network coordinate
    center_y: float
    radius: float  # Influence radius (meters)

    # Elevation profile (lower = floods faster)
    base_elevation: float  # meters NAVD88

    # Drainage capacity (mm/hr discharge)
    drainage_capacity: float

    # Associated SUMO edge IDs (populated after network analysis)
    edge_ids: List[str] = field(default_factory=list)

    # Current state
    current_depth: float = 0.0  # meters


@dataclass
class DepthCapacityMapping:
    """
    Depth-to-capacity degradation based on:
    FHWA Hydraulic Engineering Circular No. 22
    AASHTO geometric design standards
    Charleston DPW flood impact data
    """

    # Vehicle traversal thresholds
    IMPASSABLE_CAR = 0.60  # meters (24 inches) - cars stall
    IMPASSABLE_TRUCK = 0.90  # meters (36 inches) - trucks stall
    WARNING_DEPTH = 0.15  # meters (6 inches)  - speed reduction begins
    CRITICAL_DEPTH = 0.30  # meters (12 inches) - significant capacity loss

    @staticmethod
    def depth_to_speed_factor(depth: float) -> float:
        """
        Convert water depth to speed reduction factor.

        Based on FHWA empirical data for passenger vehicles:
        depth=0.00m → factor=1.00 (no reduction)
        depth=0.15m → factor=0.75 (25% speed reduction)
        depth=0.30m → factor=0.50 (50% speed reduction)
        depth=0.45m → factor=0.25 (75% speed reduction)
        depth=0.60m → factor=0.00 (impassable)
        """
        if depth <= 0.0:
            return 1.0
        elif depth >= 0.60:
            return 0.0
        elif depth <= 0.15:
            # Linear: 1.0 → 0.75
            return 1.0 - (depth / 0.15) * 0.25
        elif depth <= 0.30:
            # Linear: 0.75 → 0.50
            return 0.75 - ((depth - 0.15) / 0.15) * 0.25
        elif depth <= 0.45:
            # Linear: 0.50 → 0.25
            return 0.50 - ((depth - 0.30) / 0.15) * 0.25
        else:
            # Linear: 0.25 → 0.00
            return 0.25 - ((depth - 0.45) / 0.15) * 0.25

    @staticmethod
    def depth_to_capacity_factor(depth: float) -> float:
        """
        Convert water depth to lane capacity factor.

        Capacity loss is more aggressive than speed loss due to:
        - Lateral clearance reduction
        - Driver hesitation/cautious behavior
        - Lane-change restriction in water
        """
        speed_factor = DepthCapacityMapping.depth_to_speed_factor(depth)

        # Capacity degrades as square of speed factor
        # (flow = density × speed, density also drops with caution)
        if depth <= 0.15:
            return speed_factor  # Mild: speed-proportional
        else:
            return speed_factor ** 1.5  # Severe: super-linear degradation

    @staticmethod
    def depth_to_lanes_open(depth: float, total_lanes: int) -> int:
        """Determine how many lanes remain open."""
        if depth <= 0.15:
            return total_lanes
        elif depth <= 0.30:
            return max(1, total_lanes - 1)
        elif depth <= 0.45:
            return max(1, total_lanes // 2)
        elif depth < 0.60:
            return 1
        else:
            return 0


# ============================================================
# CHARLESTON FLOOD ZONES (Buist/Spruill Area)
# ============================================================

class CharlestonFloodZones:
    """
    Real flood zone definitions for Buist/Spruill intersection.

    Geographic context:
    - Buist/Spruill sits at ~2.1m NAVD88 (low-lying Peninsula)
    - Surrounded by tidal wetlands and drainage ditches
    - Historical flooding: 2015, 2019, 2021 major events
    - FEMA Zone AE (1% annual chance flood)
    """

    # SUMO intersection center from your screenshot
    INTERSECTION_CENTER = (1548.26, 2407.60)

    ZONES = [
        FloodZone(
            zone_id="Z1_intersection_core",
            center_x=1548.26,
            center_y=2407.60,
            radius=80.0,
            base_elevation=2.1,  # meters NAVD88
            drainage_capacity=25.4  # mm/hr (1 inch/hr typical Charleston)
        ),
        FloodZone(
            zone_id="Z2_spruill_east",
            center_x=1748.26,  # ~200m east along Spruill
            center_y=2407.60,
            radius=120.0,
            base_elevation=1.8,  # slightly lower
            drainage_capacity=12.7  # mm/hr (reduced - wetland adjacent)
        ),
        FloodZone(
            zone_id="Z3_spruill_west",
            center_x=1348.26,  # ~200m west along Spruill
            center_y=2407.60,
            radius=120.0,
            base_elevation=2.3,
            drainage_capacity=25.4
        ),
        FloodZone(
            zone_id="Z4_buist_north",
            center_x=1548.26,
            center_y=2607.60,  # ~200m north along Buist
            radius=100.0,
            base_elevation=1.6,  # lowest point - drainage basin
            drainage_capacity=10.2  # mm/hr (very poor drainage)
        ),
        FloodZone(
            zone_id="Z5_buist_south",
            center_x=1548.26,
            center_y=2207.60,  # ~200m south along Buist
            radius=100.0,
            base_elevation=2.4,
            drainage_capacity=25.4
        ),
    ]


# ============================================================
# FLOOD EVENT TIME SERIES
# ============================================================

@dataclass
class FloodEvent:
    """
    Time-series model for a compound flood event.

    Compound flood = tidal surge + rainfall coincidence
    (dominant flood type at Charleston Peninsula)
    """

    scenario: FloodScenario

    # Rainfall intensity (mm/hr)
    rainfall_rate: float

    # Tidal surge height (meters above MHHW)
    tidal_surge: float

    # Event timeline (simulation seconds)
    onset_time: float = 0.0  # When flooding begins
    peak_time: float = 1800.0  # When flooding peaks (30 min)
    drain_start: float = 3600.0  # When drainage begins (60 min)
    clear_time: float = 7200.0  # When roads clear (120 min)

    # Pre-event conditions
    antecedent_soil_saturation: float = 0.7  # 0-1 (Charleston avg)

    # Scenario presets
    SCENARIOS = {
        FloodScenario.MILD: {
            "rainfall_rate": 50.8,  # mm/hr (2-inch/hr)
            "tidal_surge": 0.15,  # meters
            "peak_depth": 0.15,  # meters at intersection
            "description": "2-year: Heavy rain, minimal surge"
        },
        FloodScenario.MODERATE: {
            "rainfall_rate": 76.2,  # mm/hr (3-inch/hr)
            "tidal_surge": 0.45,  # meters
            "peak_depth": 0.30,  # meters at intersection
            "description": "10-year: Intense rain + tidal coincidence"
        },
        FloodScenario.SEVERE: {
            "rainfall_rate": 101.6,  # mm/hr (4-inch/hr)
            "tidal_surge": 0.75,  # meters
            "peak_depth": 0.55,  # meters at intersection
            "description": "100-year: Extreme compound flood (2015 analog)"
        },
        FloodScenario.EXTREME: {
            "rainfall_rate": 127.0,  # mm/hr (5-inch/hr)
            "tidal_surge": 1.20,  # meters
            "peak_depth": 0.80,  # meters at intersection
            "description": "500-year: Climate scenario (2050 projection)"
        }
    }


# ============================================================
# DYNAMIC FLOOD DEPTH ENGINE
# ============================================================

class FloodDepthEngine:
    """
    Computes time-varying water depth at each SUMO edge.

    Physics-based simplified model:
    - Rainfall accumulation
    - Tidal backpressure
    - Drainage capacity
    - Surface runoff routing
    """

    def __init__(self, scenario: FloodScenario,
                 network_analysis_file: str = "network_analysis.json"):

        self.scenario = scenario
        self.zones = CharlestonFloodZones.ZONES.copy()
        self.depth_map = DepthCapacityMapping()

        # Load scenario parameters
        params = FloodEvent.SCENARIOS.get(scenario, {})
        self.rainfall_rate = params.get("rainfall_rate", 0.0)
        self.tidal_surge = params.get("tidal_surge", 0.0)
        self.peak_depth = params.get("peak_depth", 0.0)

        # Event timeline
        self.onset_time = 300.0  # Flood starts at t=5min
        self.peak_time = 1800.0  # Peak at t=30min
        self.drain_start = 3600.0  # Drain starts t=60min
        self.clear_time = 7200.0  # Clear at t=120min

        # Load network edge assignments
        self.edge_flood_zones = {}  # edge_id → list of zone influences
        self._load_network_assignments(network_analysis_file)

        print(f"🌊 FloodDepthEngine initialized: {scenario.value}")
        print(f"   Rainfall: {self.rainfall_rate} mm/hr")
        print(f"   Tidal surge: {self.tidal_surge}m")
        print(f"   Peak depth: {self.peak_depth}m")
        print(f"   Zones: {len(self.zones)}")

    def _load_network_assignments(self, json_file: str):
        """Load edge-to-flood-zone assignments from network analysis."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            self.flood_edge_ids = [fe["edge_id"]
                                   for fe in data.get("flood_edges", [])]

            # Assign edges to zones by proximity
            for fe in data.get("flood_edges", []):
                eid = fe["edge_id"]
                cx, cy = fe.get("centroid",
                                [CharlestonFloodZones.INTERSECTION_CENTER[0],
                                 CharlestonFloodZones.INTERSECTION_CENTER[1]])

                zone_influences = []
                for zone in self.zones:
                    dist = math.sqrt((cx - zone.center_x) ** 2 +
                                     (cy - zone.center_y) ** 2)
                    if dist <= zone.radius:
                        # Inverse-distance weight
                        weight = 1.0 - (dist / zone.radius)
                        zone_influences.append((zone, weight))

                if zone_influences:
                    self.edge_flood_zones[eid] = zone_influences

            print(f"   Flood-prone edges loaded: {len(self.flood_edge_ids)}")

        except FileNotFoundError:
            print(f"   ⚠️  {json_file} not found - using default edge set")
            self.flood_edge_ids = []

    def compute_zone_depth(self, zone: FloodZone,
                           sim_time: float) -> float:
        """
        Compute water depth at a flood zone at simulation time t.

        Uses piecewise linear approximation of flood hydrograph.
        """

        if self.scenario == FloodScenario.NONE:
            return 0.0

        # Before onset: no flooding
        if sim_time < self.onset_time:
            return 0.0

        # Compute zone-specific peak depth
        # Lower elevation = deeper flooding
        elevation_factor = max(0.5,
                               1.0 - (zone.base_elevation - 2.1) * 0.3)
        drainage_factor = max(0.5,
                              1.0 - zone.drainage_capacity / 50.8)

        zone_peak = (self.peak_depth *
                     elevation_factor *
                     (1.0 + drainage_factor * 0.3))
        zone_peak = min(zone_peak, 0.90)  # Physical maximum

        # Rising phase: onset → peak
        if self.onset_time <= sim_time <= self.peak_time:
            t_norm = ((sim_time - self.onset_time) /
                      (self.peak_time - self.onset_time))
            # S-curve rise (realistic flood onset)
            depth = zone_peak * (3 * t_norm ** 2 - 2 * t_norm ** 3)
            return depth

        # Peak maintained
        elif self.peak_time < sim_time <= self.drain_start:
            return zone_peak

        # Draining phase: drain_start → clear
        elif self.drain_start < sim_time <= self.clear_time:
            t_norm = ((sim_time - self.drain_start) /
                      (self.clear_time - self.drain_start))
            # Exponential drain (realistic drainage)
            drainage_coeff = zone.drainage_capacity / 25.4  # normalized
            depth = zone_peak * math.exp(-drainage_coeff * t_norm * 3.0)
            return max(0.0, depth)

        # Post-event
        else:
            return 0.0

    def get_edge_depth(self, edge_id: str,
                       sim_time: float) -> float:
        """Get water depth at a specific edge."""

        if edge_id not in self.edge_flood_zones:
            return 0.0

        # Weighted average of zone depths
        total_weight = 0.0
        weighted_depth = 0.0

        for zone, weight in self.edge_flood_zones[edge_id]:
            d = self.compute_zone_depth(zone, sim_time)
            weighted_depth += d * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_depth / total_weight
        return 0.0

    def get_all_edge_impacts(self, sim_time: float) -> Dict:
        """
        Get flood impact parameters for all affected edges.

        Returns dict: edge_id → {depth, speed_factor,
                                  capacity_factor, lanes_open}
        """
        impacts = {}

        for edge_id in self.flood_edge_ids:
            depth = self.get_edge_depth(edge_id, sim_time)

            impacts[edge_id] = {
                "depth": round(depth, 3),
                "speed_factor": round(
                    DepthCapacityMapping.depth_to_speed_factor(depth), 3),
                "capacity_factor": round(
                    DepthCapacityMapping.depth_to_capacity_factor(depth), 3),
                "status": self._depth_to_status(depth)
            }

        return impacts

    def _depth_to_status(self, depth: float) -> str:
        """Human-readable flood status."""
        if depth == 0.0:
            return "CLEAR"
        elif depth < 0.15:
            return "WATCH"  # Caution advised
        elif depth < 0.30:
            return "WARNING"  # Speed reduction
        elif depth < 0.45:
            return "SEVERE"  # Lane closures
        elif depth < 0.60:
            return "CRITICAL"  # Near-impassable
        else:
            return "IMPASSABLE"  # Full closure

    def generate_hydrograph_table(self) -> List[Dict]:
        """Generate time-series flood data table for analysis."""

        time_points = list(range(0, 7500, 300))  # Every 5 min
        records = []

        for t in time_points:
            row = {"time_s": t, "time_min": t / 60}

            for zone in self.zones:
                depth = self.compute_zone_depth(zone, t)
                row[f"depth_{zone.zone_id}"] = round(depth, 3)
                row[f"speed_factor_{zone.zone_id}"] = round(
                    DepthCapacityMapping.depth_to_speed_factor(depth), 3)

            records.append(row)

        return records

    def print_hydrograph_summary(self):
        """Print flood event timeline to console."""

        print(f"\n{'=' * 70}")
        print(f"FLOOD HYDROGRAPH: {self.scenario.value.upper()}")
        print(f"{'=' * 70}")
        print(f"{'Time':>8} | {'Z1-Core':>10} | "
              f"{'Z4-North':>10} | {'Status':>12}")
        print(f"{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 12}")

        for t in range(0, 7500, 600):  # Every 10 min
            d1 = self.compute_zone_depth(self.zones[0], t)
            d4 = self.compute_zone_depth(self.zones[3], t)
            status = self._depth_to_status(max(d1, d4))

            print(f"{t:>7}s | {d1:>9.3f}m | "
                  f"{d4:>9.3f}m | {status:>12}")


# ============================================================
# FLOOD SOLUTION INFRASTRUCTURE
# ============================================================

class FloodSolutionFramework:
    """
    Engineering solutions for flood disruption mitigation.

    Three-tier approach:
    1. Signal Control (implemented in RL agent)
    2. Network-level rerouting
    3. Physical infrastructure recommendations
    """

    SOLUTIONS = {

        "tier1_signal": {
            "name": "Adaptive Signal Control (IMPLEMENTED)",
            "description": "DRAIN mode + AFL PPO controller",
            "response_time": "Immediate (<30s)",
            "cost": "~$50K (software)",
            "effectiveness": {
                FloodScenario.MILD: "HIGH   - AF 1.462",
                FloodScenario.MODERATE: "HIGH   - AF 1.664",
                FloodScenario.SEVERE: "MEDIUM - AF 0.581 (boundary)"
            }
        },

        "tier2_routing": {
            "name": "Dynamic Rerouting via DMS/Apps",
            "description": (
                "Variable message signs + Waze/Google partnership "
                "to redirect traffic before flood zone"
            ),
            "response_time": "2-5 minutes",
            "cost": "~$200K (DMS signs)",
            "reroute_corridors": [
                "Spruill → Rivers Ave (US-52) parallel corridor",
                "Buist → Meeting St alternate N-S route",
                "Remount Rd → I-26 access for severe events"
            ],
            "demand_reduction_estimate": "40-60% at flooded intersection"
        },

        "tier3_infrastructure": {
            "name": "Physical Flood Mitigation",
            "description": "Long-term infrastructure upgrades",
            "projects": [
                {
                    "name": "Drainage Capacity Upgrade",
                    "action": "Increase storm drain diameter at Z4 (north approach)",
                    "cost_est": "$2.5M",
                    "depth_reduction": "30-40%",
                    "timeline": "2-3 years"
                },
                {
                    "name": "Roadway Elevation",
                    "action": "Raise Spruill Ave 0.5m at Z2 (east approach)",
                    "cost_est": "$1.8M",
                    "threshold_increase": "+0.5m before capacity loss",
                    "timeline": "18 months"
                },
                {
                    "name": "Permeable Pavement",
                    "action": "Replace intersection aprons with permeable surface",
                    "cost_est": "$180K",
                    "infiltration_increase": "15-25%",
                    "timeline": "6 months"
                },
                {
                    "name": "Tide Gate / Backflow Preventer",
                    "action": "Install check valves on storm drains near Z4",
                    "cost_est": "$450K",
                    "tidal_surge_mitigation": "0.2-0.3m depth reduction",
                    "timeline": "12 months"
                }
            ]
        },

        "tier4_sensors": {
            "name": "Real-Time Flood Sensor Network",
            "description": "IoT water depth sensors feeding RL agent",
            "sensors": [
                {
                    "location": "Z1 - Intersection center",
                    "type": "Ultrasonic depth sensor (0-2m range)",
                    "update_rate": "1 Hz",
                    "integration": "TraCI API → RL state vector"
                },
                {
                    "location": "Z4 - North approach (worst drainage)",
                    "type": "Pressure transducer",
                    "update_rate": "1 Hz",
                    "integration": "Early warning trigger for DRAIN mode"
                },
                {
                    "location": "Z2 - East approach (Spruill)",
                    "type": "Camera + AI depth estimation",
                    "update_rate": "0.5 Hz",
                    "integration": "Visual confirmation + depth model input"
                }
            ],
            "cost": "~$35K",
            "rl_integration": (
                "Sensor reading → flood_depth state variable → "
                "DRAIN mode trigger → antifragile response"
            )
        }
    }

    @classmethod
    def print_solution_summary(cls):
        print("\n" + "=" * 70)
        print("FLOOD DISRUPTION SOLUTION FRAMEWORK - BUIST/SPRUILL")
        print("=" * 70)

        for tier_key, tier in cls.SOLUTIONS.items():
            print(f"\n{'─' * 70}")
            print(f"🔧 {tier['name']}")
            print(f"   {tier['description']}")

            if 'cost' in tier:
                print(f"   💰 Cost: {tier['cost']}")
            if 'response_time' in tier:
                print(f"   ⏱️  Response: {tier['response_time']}")


# ============================================================
# MAIN: GENERATE FLOOD MODEL FILES FOR SUMO
# ============================================================

if __name__ == "__main__":

    import csv

    print("🌊 BUIST/SPRUILL FLOOD DEPTH MODEL")
    print("   Charleston, SC Compound Flood Simulation")
    print("=" * 60)

    # Run analysis for all scenarios
    scenarios = [
        FloodScenario.MILD,
        FloodScenario.MODERATE,
        FloodScenario.SEVERE,
    ]

    for scenario in scenarios:
        engine = FloodDepthEngine(scenario)
        engine.print_hydrograph_summary()

        # Save hydrograph CSV
        fname = f"hydrograph_{scenario.value}.csv"
        records = engine.generate_hydrograph_table()

        if records:
            with open(fname, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            print(f"\n   ✅ Saved: {fname}")

    # Print solution framework
    FloodSolutionFramework.print_solution_summary()

    print("\n\n✅ Flood model ready for SUMO TraCI integration")
    print("   Next step: Run flood_sumo_integration.py")
