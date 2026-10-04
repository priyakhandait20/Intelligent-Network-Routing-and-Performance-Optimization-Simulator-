"""
Step 2: Traffic Generation (standalone demo)
Run: python traffic_demo.py
"""
import random
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_NODES = 12
FLOWS_PER_STEP = 60
HOTSPOTS = [0, 1]        # in routers pe zyada traffic jaata hai
HOTSPOT_PROB = 0.8       # 40% flows hotspot ki taraf


def generate_flows(rng, n=N_NODES, k=FLOWS_PER_STEP):
    flows = []
    for _ in range(k):
        # 1. destination chuno (hotspot ya random)
        if rng.random() < HOTSPOT_PROB:
            dst = rng.choice(HOTSPOTS)
        else:
            dst = rng.randrange(n)
        # 2. source chuno (destination se alag)
        src = rng.randrange(n)
        while src == dst:
            src = rng.randrange(n)
        # 3. demand chuno (5 se 20 units)
        demand = rng.uniform(5, 20)
        flows.append((src, dst, demand))
    return flows


if __name__ == "__main__":
    rng = random.Random(43)          # fixed seed = har baar same traffic
    flows = generate_flows(rng)

    print(f"{'Flow':<6}{'Source':<9}{'Dest':<8}{'Demand':<8}")
    print("-" * 31)
    for i, (s, d, dem) in enumerate(flows, 1):
        print(f"{i:<6}{s:<9}{d:<8}{dem:<8.1f}")

    total = sum(f[2] for f in flows)
    to_hot = sum(1 for f in flows if f[1] in HOTSPOTS)
    print(f"\nTotal traffic: {total:.1f} units")
    print(f"Hotspot ki taraf flows: {to_hot} / {len(flows)}")

    # graph: har router pe kitne flows aa rahe hain
    counts = Counter(f[1] for f in flows)
    plt.bar(range(N_NODES), [counts.get(i, 0) for i in range(N_NODES)])
    plt.xlabel("Destination router")
    plt.ylabel("Incoming flows")
    plt.title("Traffic per destination (routers 0 & 1 = hotspots)")
    plt.xticks(range(N_NODES))
    plt.savefig("traffic.png", dpi=150)
    print("Graph saved: traffic.png")