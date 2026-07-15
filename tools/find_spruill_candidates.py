# find_spruill_candidates.py

import xml.etree.ElementTree as ET
from pathlib import Path
import math

NET_FILE = Path("buist_spruill_real.net.xml")

FLOOD_CENTER = (1548.26, 2407.60)

tree = ET.parse(NET_FILE)
root = tree.getroot()

candidates = []

for edge in root.findall("edge"):
    edge_id = edge.get("id")

    if not edge_id or edge_id.startswith(":"):
        continue

    lanes = edge.findall("lane")
    if not lanes:
        continue

    lane = lanes[0]
    length = float(lane.get("length", 0.0))
    speed = float(lane.get("speed", 0.0))
    shape = lane.get("shape", "")

    if not shape:
        continue

    coords = []
    for pt in shape.split():
        x, y = pt.split(",")[:2]
        coords.append((float(x), float(y)))

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]

    centroid_x = sum(xs) / len(xs)
    centroid_y = sum(ys) / len(ys)

    dist = math.sqrt((centroid_x - FLOOD_CENTER[0])**2 + (centroid_y - FLOOD_CENTER[1])**2)

    # likely Spruill: mostly vertical edges near intersection
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)

    if dist <= 180 and dy > dx and length > 10:
        candidates.append((edge_id, dist, length, speed, centroid_x, centroid_y, dx, dy))

candidates.sort(key=lambda x: x[1])

print("\nLikely Spruill Avenue North/South Approach Edges")
print("=" * 90)

for eid, dist, length, speed, cx, cy, dx, dy in candidates[:50]:
    print(f"{eid:30s} dist={dist:6.1f} m | len={length:7.1f} m | speed={speed:5.1f} | centroid=({cx:.1f},{cy:.1f})")