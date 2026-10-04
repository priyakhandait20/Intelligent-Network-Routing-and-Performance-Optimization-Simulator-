"""
Network + Source-to-Destination path visualizer
Run: python network_viz.py
Output: network.png (topology) and path_compare.png (source -> destination paths)
"""
import random
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

SEED = 42
N_NODES = 12
HOTSPOTS = [0, 1]
FLOWS = 30


# ---------- same network as simulation.py ----------
def build_topology():
    G = nx.connected_watts_strogatz_graph(N_NODES, 4, 0.3, seed=SEED)
    rng = random.Random(SEED)
    for u, v in G.edges():
        G[u][v]["delay"] = rng.uniform(1, 5)
        G[u][v]["capacity"] = rng.choice([60, 100, 150])
        G[u][v]["load"] = 0.0
    return G


def util(e):
    return e["load"] / e["capacity"]


def cost(e):
    return e["delay"] * (1 + 4 * min(util(e), 2.0) ** 2)


def path_delay(G, p):
    return sum(cost(G[u][v]) for u, v in zip(p, p[1:]))


def make_traffic(rng):
    flows = []
    for _ in range(FLOWS):
        dst = rng.choice(HOTSPOTS) if rng.random() < 0.4 else rng.randrange(N_NODES)
        src = rng.randrange(N_NODES)
        while src == dst:
            src = rng.randrange(N_NODES)
        flows.append((src, dst, rng.uniform(5, 20)))
    return flows


# ---------- drawing helpers ----------
def node_colors(G, src=None, dst=None):
    cols = []
    for n in G.nodes():
        if n == src:
            cols.append("#2ca02c")        # source = green
        elif n == dst:
            cols.append("#d62728")        # destination = red
        elif n in HOTSPOTS:
            cols.append("#ff9f1c")        # hotspot = orange
        else:
            cols.append("#dddddd")
    return cols


def draw_path(ax, G, pos, path, title, src, dst):
    edges = list(G.edges())
    utils = [util(G[u][v]) for u, v in edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=utils,
                           edge_cmap=plt.cm.YlOrRd, edge_vmin=0, edge_vmax=1.5, width=2.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(zip(path, path[1:])),
                           edge_color="#1f77b4", width=6, alpha=0.85)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors(G, src, dst),
                           node_size=650, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight="bold")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main():
    G = build_topology()
    pos = nx.circular_layout(G)

    # ---------- Figure 1: plain network ----------
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, edge_color="#888888")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors(G),
                           node_size=800, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight="bold")
    labels = {(u, v): f"{G[u][v]['delay']:.1f}ms/{G[u][v]['capacity']}"
              for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=7)
    ax.set_title("Network: 12 routers, 24 links\n"
                 "(label = delay/capacity, orange = hotspot routers)")
    ax.axis("off")
    plt.savefig("network.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------- traffic ko static routing se bhejo -> load banta hai ----------
    for s, d, dem in make_traffic(random.Random(SEED + 1)):
        p = nx.shortest_path(G, s, d, weight="delay")
        for u, v in zip(p, p[1:]):
            G[u][v]["load"] += dem

    # ---------- aisa source-destination dhundo jahan dono paths alag hon ----------
    best = None
    for s in G.nodes():
        for d in G.nodes():
            if s == d:
                continue
            p_static = nx.shortest_path(G, s, d, weight="delay")
            p_adapt = nx.shortest_path(G, s, d, weight=lambda u, v, e: cost(e))
            gain = path_delay(G, p_static) - path_delay(G, p_adapt)
            if p_static != p_adapt and (best is None or gain > best[0]):
                best = (gain, s, d, p_static, p_adapt)

    if best is None:
        print("Is traffic par dono path same hain. SEED ya FLOWS badal ke try karo.")
        return
    gain, s, d, p_static, p_adapt = best

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_path(axes[0], G, pos, p_static,
              f"Static Dijkstra: {' -> '.join(map(str, p_static))}\n"
              f"Delay = {path_delay(G, p_static):.1f} ms", s, d)
    draw_path(axes[1], G, pos, p_adapt,
              f"Adaptive / Smart: {' -> '.join(map(str, p_adapt))}\n"
              f"Delay = {path_delay(G, p_adapt):.1f} ms", s, d)
    fig.suptitle(f"Source {s} (green) to Destination {d} (red)  |  "
                 "blue = chosen path, link colour = congestion", fontsize=13)
    sm = ScalarMappable(norm=Normalize(0, 1.5), cmap=plt.cm.YlOrRd)
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Link utilization (1.0 = full)")
    plt.savefig("path_compare.png", dpi=150, bbox_inches="tight")

    print(f"Source = {s}, Destination = {d}")
    print(f"Static path  : {p_static}  delay = {path_delay(G, p_static):.1f} ms")
    print(f"Smart path   : {p_adapt}  delay = {path_delay(G, p_adapt):.1f} ms")
    print("Saved: network.png, path_compare.png")


if __name__ == "__main__":
    main()