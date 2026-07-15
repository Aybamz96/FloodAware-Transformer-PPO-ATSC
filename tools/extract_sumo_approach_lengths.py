# extract_approach_lengths_final.py

import xml.etree.ElementTree as ET
from pathlib import Path

NET_FILE = Path("buist_spruill_real.net.xml")

APPROACH_EDGES = {
    "Westbound / Buist Ave": [
        "-12154541#4",
        "-12154541#2",
    ],
    "Eastbound / Buist Ave": [
        "12154541#4",
        "12154541#2",
    ],
    # Add these after confirming Spruill IDs from network_analysis.json
    "Northbound / Spruill Ave": [],
    "Southbound / Spruill Ave": [],
}

tree = ET.parse(NET_FILE)
root = tree.getroot()

edge_lengths = {}

for edge in root.findall("edge"):
    edge_id = edge.get("id")

    if not edge_id or edge_id.startswith(":"):
        continue

    lanes = edge.findall("lane")
    if not lanes:
        continue

    length = float(lanes[0].get("length", 0.0))
    speed = float(lanes[0].get("speed", 0.0))

    edge_lengths[edge_id] = {
        "length": length,
        "speed": speed,
    }

print("\nApproach Length Summary")
print("=" * 65)

for direction, edges in APPROACH_EDGES.items():
    total = 0.0

    print(f"\n{direction}")
    print("-" * 65)

    for eid in edges:
        if eid in edge_lengths:
            length = edge_lengths[eid]["length"]
            speed = edge_lengths[eid]["speed"]
            total += length
            print(f"{eid:25s} length = {length:7.2f} m | speed = {speed:5.2f} m/s")
        else:
            print(f"{eid:25s} NOT FOUND")

    print(f"Total approach length: {total:.2f} m")