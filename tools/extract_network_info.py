# extract_network_info.py
"""
Extract edge and junction information from buist_spruill_real.net.xml
to map flood zones to real SUMO edge IDs.
"""

import xml.etree.ElementTree as ET
import json
import math


def extract_network_info(net_file: str) -> dict:
    """Parse SUMO network file and extract edge/junction data."""

    tree = ET.parse(net_file)
    root = tree.getroot()

    network_info = {
        "edges": {},
        "junctions": {},
        "tls": [],
        "bounds": {}
    }

    # --- Extract location/projection info ---
    location = root.find('location')
    if location is not None:
        network_info["projection"] = location.get('projParameter', '')
        offset = location.get('netOffset', '0,0').split(',')
        network_info["net_offset"] = {
            "x": float(offset[0]),
            "y": float(offset[1])
        }
        orig_boundary = location.get('origBoundary', '')
        network_info["orig_boundary"] = orig_boundary

        conv_boundary = location.get('convBoundary', '').split(',')
        if len(conv_boundary) == 4:
            network_info["bounds"] = {
                "xmin": float(conv_boundary[0]),
                "ymin": float(conv_boundary[1]),
                "xmax": float(conv_boundary[2]),
                "ymax": float(conv_boundary[3])
            }

    # --- Extract edges ---
    for edge in root.findall('edge'):
        edge_id = edge.get('id')

        # Skip internal edges
        if edge_id.startswith(':'):
            continue

        from_node = edge.get('from', '')
        to_node = edge.get('to', '')
        edge_type = edge.get('type', '')

        lanes = []
        for lane in edge.findall('lane'):
            lane_data = {
                "id": lane.get('id'),
                "index": int(lane.get('index', 0)),
                "speed": float(lane.get('speed', 13.89)),
                "length": float(lane.get('length', 0)),
            }
            # Extract shape for geographic mapping
            shape = lane.get('shape', '')
            if shape:
                coords = []
                for pt in shape.split():
                    xy = pt.split(',')
                    if len(xy) >= 2:
                        coords.append((float(xy[0]), float(xy[1])))
                lane_data["shape"] = coords

                # Compute centroid for flood zone mapping
                if coords:
                    cx = sum(c[0] for c in coords) / len(coords)
                    cy = sum(c[1] for c in coords) / len(coords)
                    lane_data["centroid"] = (cx, cy)

            lanes.append(lane_data)

        network_info["edges"][edge_id] = {
            "from": from_node,
            "to": to_node,
            "type": edge_type,
            "lanes": lanes,
            "n_lanes": len(lanes)
        }

    # --- Extract junctions ---
    for junc in root.findall('junction'):
        junc_id = junc.get('id')
        junc_type = junc.get('type', '')

        if junc_id.startswith(':'):
            continue

        network_info["junctions"][junc_id] = {
            "type": junc_type,
            "x": float(junc.get('x', 0)),
            "y": float(junc.get('y', 0)),
        }

    # --- Extract TLS programs ---
    for tls in root.findall('tlLogic'):
        tls_info = {
            "id": tls.get('id'),
            "type": tls.get('type'),
            "programID": tls.get('programID'),
            "phases": []
        }
        for phase in tls.findall('phase'):
            tls_info["phases"].append({
                "duration": float(phase.get('duration', 30)),
                "state": phase.get('state', '')
            })
        network_info["tls"].append(tls_info)

    return network_info


def find_flood_prone_edges(network_info: dict,
                           flood_center_sumo: tuple = (1548.0, 2407.0),
                           radius: float = 200.0) -> list:
    """
    Identify edges within flood zone radius.

    flood_center_sumo: SUMO coordinate of intersection center
                       (from your screenshot: x=1548.26, y=2407.60)
    radius: search radius in meters
    """

    flood_edges = []
    cx, cy = flood_center_sumo

    for edge_id, edge_data in network_info["edges"].items():
        for lane in edge_data["lanes"]:
            if "centroid" in lane:
                lx, ly = lane["centroid"]
                dist = math.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)

                if dist <= radius:
                    flood_edges.append({
                        "edge_id": edge_id,
                        "lane_id": lane["id"],
                        "distance": round(dist, 2),
                        "centroid": lane["centroid"],
                        "speed": lane["speed"],
                        "length": lane["length"]
                    })
                    break  # One entry per edge

    # Sort by distance from flood center
    flood_edges.sort(key=lambda x: x["distance"])

    return flood_edges


if __name__ == "__main__":

    NET_FILE = "buist_spruill_real.net.xml"

    print("=" * 60)
    print("BUIST/SPRUILL NETWORK ANALYSIS")
    print("=" * 60)

    info = extract_network_info(NET_FILE)

    print(f"\n📊 Network Statistics:")
    print(f"   Total edges:     {len(info['edges'])}")
    print(f"   Total junctions: {len(info['junctions'])}")
    print(f"   TLS controllers: {len(info['tls'])}")

    if info["bounds"]:
        b = info["bounds"]
        print(f"\n📍 Network Bounds (SUMO coords):")
        print(f"   X: {b['xmin']:.1f} → {b['xmax']:.1f}")
        print(f"   Y: {b['ymin']:.1f} → {b['ymax']:.1f}")

    print(f"\n🚦 Traffic Light Systems:")
    for tls in info["tls"]:
        total_cycle = sum(p["duration"] for p in tls["phases"])
        print(f"   TLS '{tls['id']}': "
              f"{len(tls['phases'])} phases, "
              f"cycle={total_cycle:.0f}s")

    # Find flood-prone edges
    # SUMO coord from your screenshot: x=1548.26, y=2407.60
    FLOOD_CENTER = (1548.26, 2407.60)

    print(f"\n🌊 Flood-Prone Edges (within 200m of intersection):")
    flood_edges = find_flood_prone_edges(info, FLOOD_CENTER, radius=200.0)

    for fe in flood_edges[:20]:  # Show top 20 closest
        print(f"   {fe['edge_id']:30s} | "
              f"dist={fe['distance']:6.1f}m | "
              f"speed={fe['speed']:.1f}m/s | "
              f"len={fe['length']:.1f}m")

    # Save full results
    output = {
        "network_stats": {
            "n_edges": len(info["edges"]),
            "n_junctions": len(info["junctions"]),
            "n_tls": len(info["tls"]),
            "bounds": info["bounds"]
        },
        "tls_info": info["tls"],
        "flood_edges": flood_edges,
        "all_edge_ids": list(info["edges"].keys())
    }

    with open("network_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Full analysis saved to network_analysis.json")
    print(f"   ({len(flood_edges)} flood-prone edges identified)")
