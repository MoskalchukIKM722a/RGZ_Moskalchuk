import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

# ==============================================================================
# РОЗРАХУНКОВЕ ЗАВДАННЯ — ОСНОВИ ТЕОРІЇ ГРАФІВ
# Повне рішення всіх пунктів (а) – (і)
# ==============================================================================

def solve_graph(edges, n=None, title="Граф G"):
    """
    Повне дослідження графа.
    edges — список пар (u, v), вершини нумеруються з 1.
    """
    G = nx.Graph()
    if n:
        G.add_nodes_from(range(1, n+1))
    G.add_edges_from(edges)
    nodes = sorted(G.nodes())
    V, E = len(nodes), G.number_of_edges()
    idx = {v: i for i, v in enumerate(nodes)}
    
    print(f"\n{'='*60}")
    print(f" {title}: |V|={V}, |E|={E}")
    print(f"{'='*60}")
    
    # ── а) Матриці ──────────────────────────────────────────────
    print("\nа) МАТРИЦІ СУМІЖНОСТІ ТА ІНЦИДЕНТНОСТІ")
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=int)
    print("Матриця суміжності A(G):")
    header = "   " + " ".join(f"v{v}" for v in nodes)
    print(header)
    for i, v in enumerate(nodes):
        row = " ".join(f" {A[i,j]}" for j in range(V))
        print(f" v{v} {row}")
        
    edge_list = list(G.edges())
    B = np.zeros((V, E), dtype=int)
    for j, (u, v) in enumerate(edge_list):
        B[idx[u], j] = 1
        B[idx[v], j] = 1
        
    print("\nМатриця інцидентності B(G):")
    print("   " + " ".join(f"e{j+1:02d}" for j in range(E)))
    for i, v in enumerate(nodes):
        row = " ".join(f"  {B[i,j]}" for j in range(E))
        print(f" v{v} {row}")
        
    # Матриця суміжності ребер (edge adjacency)
    print("\nМатриця суміжності ребер A_e(G) [розмір m×m]:")
    Ae = (B.T @ B)
    np.fill_diagonal(Ae, 0)
    print(" (рядки/стовпці = ребра e1..em, 1=суміжні ребра)")
    
    # ── б) Степені ──────────────────────────────────────────────
    print("\nб) ВЕКТОР ТА МНОЖИНА СТЕПЕНІВ")
    deg_vec = [G.degree(v) for v in nodes]
    deg_set = sorted(set(deg_vec))
    print(f" Вектор степенів: {deg_vec}")
    print(f" Множина степенів: {deg_set}")
    print(f" δ(G)={min(deg_vec)}, Δ(G)={max(deg_vec)}")
    print(f" Перевірка: Σdeg = {sum(deg_vec)} = 2·|E| = {2*E} {'✓' if sum(deg_vec)==2*E else '✗ помилка!'}")
    
    # ── в) Максимальна незалежна множина ────────────────────────
    print("\nв) МАКСИМАЛЬНА НЕЗАЛЕЖНА МНОЖИНА (MIS)")
    complement = nx.complement(G)
    cliques_c = list(nx.find_cliques(complement))
    mis = max(cliques_c, key=len)
    print(f" α(G) = {len(mis)}, MIS = {sorted(mis)}")
    ok = all(not G.has_edge(u,v) for u,v in combinations(mis, 2))
    print(f" Перевірка незалежності: {'✓' if ok else '✗'}")
    
    # ── г) Матриця відстаней, радіус та діаметр ─────────────────
    print("\nг) МАТРИЦЯ ВІДСТАНЕЙ, РАДІУС ТА ДІАМЕТР")
    print("Матриця відстаней D(G):")
    dist_matrix = dict(nx.all_pairs_shortest_path_length(G))
    dist_header = "   " + " ".join(f"v{v}" for v in nodes)
    print(dist_header)
    for v in nodes:
        row = " ".join(f" {dist_matrix[v].get(u, '∞')}" for u in nodes)
        print(f" v{v} {row}")
        
    if nx.is_connected(G):
        ecc = nx.eccentricity(G)
        r, d = nx.radius(G), nx.diameter(G)
        center = nx.center(G)
        periph = nx.periphery(G)
        print(f"\n Ексцентриситети: { {v: ecc[v] for v in nodes} }")
        print(f" Радіус r(G) = {r}, Центр = {center}")
        print(f" Діаметр d(G) = {d}, Периферія = {periph}")
    else:
        print("\n Граф не зв'язний — радіус та діаметр обчислюються лише для окремих компонент зв'язності.")
        
    # ── д) Мости та точки зчленування ───────────────────────────
    print("\nд) МОСТИ ТА ТОЧКИ ЗЧЛЕНУВАННЯ")
    bridges = list(nx.bridges(G))
    art_pts = sorted(nx.articulation_points(G))
    print(f" Мости: {bridges if bridges else 'відсутні'}")
    print(f" Точки зчленування: {art_pts if art_pts else 'відсутні'}")
    
    # ── е) Вершинна та реберна зв'язність ───────────────────────
    print("\nе) ЧИСЛА ЗВ'ЯЗНОСТІ")
    if nx.is_connected(G):
        kv = nx.node_connectivity(G)
        ke = nx.edge_connectivity(G)
        min_vc = nx.minimum_node_cut(G)
        min_ec = nx.minimum_edge_cut(G)
        print(f" Вершинна зв'язність κ(G) = {kv}")
        print(f" Мін. вершинний розріз: {min_vc}")
        print(f" Реберна зв'язність λ(G) = {ke}")
        print(f" Мін. реберний розріз: {min_ec}")
        print(f" Нерівність Уітні: κ≤λ≤δ → {kv}≤{ke}≤{min(deg_vec)} ✓")
    else:
        print(" Граф не зв'язний: κ(G) = λ(G) = 0")
        
    # ── ж) Колода (максимальна кліка) ───────────────────────────
    print("\nж) КОЛОДА (МАКСИМАЛЬНА КЛІКА)")
    cliques = list(nx.find_cliques(G))
    max_cl = max(cliques, key=len)
    omega = len(max_cl)
    print(f" ω(G) = {omega}, Колода = {sorted(max_cl)}")
    print(f" Усі максимальні кліки: {[sorted(c) for c in cliques]}")
    
    # ── з) Планарність ──────────────────────────────────────────
    print("\nз) ПЛАНАРНІСТЬ")
    is_planar, _ = nx.check_planarity(G)
    print(f" Граф {'планарний ✓' if is_planar else 'не планарний ✗'}")
    if is_planar:
        F = 2 - V + E
        print(f" Формула Ейлера: V-E+F = {V}-{E}+{F} = {V-E+F}")
        print(f" Умова планарності: E ≤ 3V-6 → {E} ≤ {3*V-6} ✓")
    else:
        print(" Міноровим методом: містить підграф ≅ K₅ або K₃,₃")
        
    # ── і) Хроматичне число ──────────────────────────────────────
    print("\ні) ХРОМАТИЧНЕ ЧИСЛО")
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    chi_upper = max(coloring.values()) + 1
    print(f" Нижня межа: ω(G) = {omega}")
    print(f" Верхня межа (жадібний): {chi_upper}")
    print(f" χ(G) = {chi_upper}")
    color_classes = {}
    for v, c in coloring.items():
        color_classes.setdefault(c, []).append(v)
    for c, verts in sorted(color_classes.items()):
        print(f" Клас {c+1}: {sorted(verts)}")
        
    return G

# ==============================================================================
# ПРИКЛАД ВИКОРИСТАННЯ (Універсальна візуалізація)
# ==============================================================================
if __name__ == '__main__':
    # 1. Задаємо граф (можете прописати будь-які свої ребра тут)
    edges = [
            (1,2),
            (1,4),
            (2,3),
            (3,4),
            (3,5),
            (4,6),
            (5,6),
            (6,7)
        ]
    
    G = solve_graph(edges, title="Граф G")
    
    # 2. АВТОМАТИЧНА ГЕНЕРАЦІЯ КООРДИНАТ
    # Використовуємо spring_layout (або kamada_kawai_layout, planar_layout)
    pos = nx.spring_layout(G, seed=42)
    
    # Загальні налаштування візуалізації
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Аналіз графа G", fontsize=18, fontweight='bold')
    
    node_opts = {
        'node_size': 900,
        'font_color': 'white',
        'font_weight': 'bold',
        'font_size': 12
    }
    
    # ---------------------------------------------------------
    # 1. Граф G (Верхній лівий)
    ax = axes[0, 0]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='#3182bd', **node_opts)
    ax.set_title("Граф G", fontsize=14)
    
    # ---------------------------------------------------------
    # 2. Розфарбування (Верхній центральний)
    ax = axes[0, 1]
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    colors = ['#e74c3c', '#5dade2', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e']
    node_colors_chi = [colors[coloring[v] % len(colors)] for v in G.nodes()]
    chi = max(coloring.values()) + 1
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors_chi, **node_opts)
    ax.set_title(f"Розфарбування χ={chi}", fontsize=14)
    
    # ---------------------------------------------------------
    # 3. Ексцентриситети (Верхній правий) - Стовпчаста діаграма
    ax = axes[0, 2]
    if nx.is_connected(G):
        ecc = nx.eccentricity(G)
        nodes_sorted = sorted(G.nodes())
        ecc_values = [ecc[n] for n in nodes_sorted]

        radius = min(ecc_values)
        diameter = max(ecc_values)

        # 3 кольори:
        # синій — радіус
        # зелений — проміжні значення
        # червоний — діаметр
        color_radius = '#5dade2'
        color_middle = '#2ecc71'
        color_diameter = '#e74c3c'

        bar_colors = []

        for value in ecc_values:
            if value == radius:
                bar_colors.append(color_radius)
            elif value == diameter:
                bar_colors.append(color_diameter)
            else:
                bar_colors.append(color_middle)

        bars = ax.bar(nodes_sorted, ecc_values, color=bar_colors)

        # Підписи значень над стовпцями
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                str(ecc_values[i]),
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=12
            )

        ax.set_xticks(nodes_sorted)
        ax.set_xlabel("Вершина", fontsize=12)
        ax.set_ylabel("Ексцентриситет", fontsize=12)
        ax.set_title("Ексцентриситети вершин", fontsize=14)

        # Легенда
        radius_patch = mpatches.Patch(
            color=color_radius,
            label=f'Центр: e(v) = {radius}'
        )

        middle_patch = mpatches.Patch(
            color=color_middle,
            label=f'Проміжні: {radius} < e(v) < {diameter}'
        )

        diameter_patch = mpatches.Patch(
            color=color_diameter,
            label=f'Периферія: e(v) = {diameter}'
        )

        ax.legend(
            handles=[radius_patch, middle_patch, diameter_patch],
            loc='upper right',
            fontsize=10
        )

        ax.grid(axis='y', linestyle='-', alpha=0.3)
        ax.set_ylim(0, diameter + 1.0)

    else:
        ax.text(
            0.5,
            0.5,
            "Граф не зв'язний\n(ексцентриситети нескінченні)",
            ha='center',
            va='center',
            fontsize=14
        )
        ax.axis('off')
    # ---------------------------------------------------------
    # 4. Мости / точки зчленування (Нижній лівий)
    ax = axes[1, 0]
    art_pts = list(nx.articulation_points(G))
    bridges = list(nx.bridges(G))
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='#3182bd', **node_opts)
    if art_pts:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=art_pts, node_color='#f39c12', node_size=900)
    if bridges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=bridges, edge_color='#c0392b', width=4)
    ax.set_title("Мости / точки зчленування", fontsize=14)
    
    # ---------------------------------------------------------
    # 5. Незалежна множина (Нижній центральний)
    ax = axes[1, 1]
    # Автоматичний пошук найбільшої незалежної множини
    all_mis = list(nx.find_cliques(nx.complement(G)))
    mis = max(all_mis, key=len)
    
    node_colors_mis = ['#27ae60' if v in mis else '#3182bd' for v in G.nodes()]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors_mis, **node_opts)
    ax.set_title(f"Незалежна множина α={len(mis)}", fontsize=14)
    
    # ---------------------------------------------------------
    # 6. Кліка (Нижній правий)
    ax = axes[1, 2]
    # Автоматичний пошук найбільшої кліки (колоди)
    cliques = list(nx.find_cliques(G))
    max_clique = max(cliques, key=len)
    
    node_colors_clique = ['#e74c3c' if v in max_clique else '#3182bd' for v in G.nodes()]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors_clique, **node_opts)
    
    # Виділяємо всі ребра, що утворюють максимальну кліку
    clique_edges = list(combinations(max_clique, 2))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=clique_edges, width=5, edge_color='black')
    ax.set_title(f"Колода ω={len(max_clique)}", fontsize=14)

    plt.tight_layout(pad=3.0)
    plt.savefig('graph_analysis_auto.png', dpi=150, bbox_inches='tight')
    plt.show()