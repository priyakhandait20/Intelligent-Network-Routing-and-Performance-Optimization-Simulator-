"""
Intelligent Network Routing and Performance Optimization Simulator
------------------------------------------------------------------
Compares 3 routing strategies on the same topology + same traffic:
  1. Static Dijkstra    -> shortest path by base delay (traditional)
  2. Adaptive Dijkstra  -> shortest path by *current* congestion-aware delay
  3. Q-Routing (RL)     -> learns best next-hop per destination (intelligent)

Run:
    pip install networkx matplotlib numpy
    python simulator.py
Output: results table in terminal + results.png
"""
import random
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")          # file me save karne ke liye (GUI nahi chahiye)
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
SEED = 42
N_NODES = 12
N_STEPS = 400            # simulation time steps
FLOWS_PER_STEP = 30      # traffic load per step
HOTSPOTS = [0, 1]        # in nodes pe zyada traffic jaata hai (congestion banane ke liye)
HOTSPOT_PROB = 0.4
ALPHA = 0.3              # Q-learning rate
EPS_START, EPS_END, EPS_DECAY = 0.3, 0.02, 0.99
MAX_HOPS = 20


# ---------------- TOPOLOGY ----------------
def build_topology(n=N_NODES, seed=SEED):
    G = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=seed)
    rng = random.Random(seed)
    for u, v in G.edges():
        G[u][v]["delay"] = rng.uniform(1, 5)               # base delay (ms)
        G[u][v]["capacity"] = rng.choice([60, 100, 150])   # link capacity
        G[u][v]["load"] = 0.0
    return G


def link_delay(G, u, v, extra=0.0):
    """Delay badhta hai jaise-jaise link congested hoti hai."""
    e = G[u][v]
    util = min((e["load"] + extra) / e["capacity"], 2.0)
    return e["delay"] * (1 + 4 * util ** 2)


# ---------------- TRAFFIC ----------------
def generate_flows(rng, n=N_NODES, k=FLOWS_PER_STEP):
    flows = []
    for _ in range(k):
        dst = rng.choice(HOTSPOTS) if rng.random() < HOTSPOT_PROB else rng.randrange(n)
        src = rng.randrange(n)
        while src == dst:
            src = rng.randrange(n)
        flows.append((src, dst, rng.uniform(5, 20)))   # (src, dst, demand)
    return flows


# ---------------- ROUTING ALGORITHMS ----------------
def static_dijkstra(G, s, d):
    return nx.shortest_path(G, s, d, weight="delay")


def adaptive_dijkstra(G, s, d):
    def w(u, v, data):
        util = min(data["load"] / data["capacity"], 2.0)
        return data["delay"] * (1 + 4 * util ** 2)
    return nx.shortest_path(G, s, d, weight=w)


class QRouter:
    """Q-routing: Q[x][d][y] = estimated delivery time from x to d via neighbour y."""

    def __init__(self, G, rng):
        self.G, self.rng = G, rng
        spl = dict(nx.all_pairs_dijkstra_path_length(G, weight="delay"))
        self.Q = {
            x: {d: {y: G[x][y]["delay"] + spl[y][d] for y in G[x]} for d in G}
            for x in G
        }

    def route(self, s, d, demand, eps):
        G, Q = self.G, self.Q
        path, visited, x = [s], {s}, s
        while x != d and len(path) < MAX_HOPS:
            nbrs = [y for y in G[x] if y not in visited] or list(G[x])
            if self.rng.random() < eps:                      # explore
                y = self.rng.choice(nbrs)
            else:                                            # exploit
                y = min(nbrs, key=lambda n: Q[x][d][n])
            t = link_delay(G, x, y, extra=demand)
            future = 0.0 if y == d else min(Q[y][d].values())
            Q[x][d][y] += ALPHA * (t + future - Q[x][d][y])  # Q-learning update
            path.append(y)
            visited.add(y)
            x = y
        if path[-1] != d:                                    # safety fallback
            path = static_dijkstra(G, s, d)
        return path


# ---------------- METRICS ----------------
def evaluate(G, flows, paths):
    total_delay, delivered, offered = 0.0, 0.0, 0.0
    for (s, d, dem), p in zip(flows, paths):
        surv = 1.0
        for u, v in zip(p, p[1:]):
            e = G[u][v]
            total_delay += link_delay(G, u, v)
            util = e["load"] / e["capacity"]
            if util > 1:
                surv *= 1 / util                             # overload => packet drop
        delivered += dem * surv
        offered += dem
    max_util = max(e["load"] / e["capacity"] for _, _, e in G.edges(data=True))
    return {
        "delay": total_delay / len(flows),
        "loss": 100 * (1 - delivered / offered),
        "throughput": delivered,
        "max_util": max_util,
    }


# ---------------- SIMULATION ----------------
def run(algo):
    G = build_topology()
    traffic_rng = random.Random(SEED + 1)     # same traffic for all algorithms
    algo_rng = random.Random(SEED + 2)
    qr = QRouter(G, algo_rng) if algo == "Q-Routing" else None
    history = []

    for step in range(N_STEPS):
        for _, _, e in G.edges(data=True):
            e["load"] = 0.0
        flows = generate_flows(traffic_rng)
        eps = max(EPS_END, EPS_START * EPS_DECAY ** step)
        paths = []
        for s, d, dem in flows:
            if algo == "Static Dijkstra":
                p = static_dijkstra(G, s, d)
            elif algo == "Adaptive Dijkstra":
                p = adaptive_dijkstra(G, s, d)
            else:
                p = qr.route(s, d, dem, eps)
            for u, v in zip(p, p[1:]):
                G[u][v]["load"] += dem
            paths.append(p)
        history.append(evaluate(G, flows, paths))
    return history


def smooth(x, w=15):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    algos = ["Static Dijkstra", "Adaptive Dijkstra", "Q-Routing"]
    results = {a: run(a) for a in algos}

    # ---- summary table (last 100 steps = after learning) ----
    print(f"\n{'Algorithm':<20}{'Delay(ms)':>11}{'Loss(%)':>10}{'Thru':>9}{'MaxUtil':>9}")
    print("-" * 59)
    for a in algos:
        tail = results[a][-100:]
        m = {k: np.mean([h[k] for h in tail]) for k in tail[0]}
        print(f"{a:<20}{m['delay']:>11.2f}{m['loss']:>10.2f}"
              f"{m['throughput']:>9.1f}{m['max_util']:>9.2f}")

    # ---- plots ----
    titles = {"delay": "Avg End-to-End Delay (ms)", "loss": "Packet Loss (%)",
              "throughput": "Throughput", "max_util": "Max Link Utilization"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key in zip(axes.flat, titles):
        for a in algos:
            ax.plot(smooth([h[key] for h in results[a]]), label=a)
        ax.set_title(titles[key])
        ax.set_xlabel("Time step")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nGraphs saved: results.png")


if __name__ == "__main__":
    main()