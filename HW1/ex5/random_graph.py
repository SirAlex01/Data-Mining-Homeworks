import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import factorial
import itertools

# Funzione per creare un grafo Erdős-Rényi
def erdos_renyi_graph(n, p):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j)
    
    return G

# Funzione per visualizzare il grafo
def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', font_size=15, font_color='black')
    plt.show()

# Funzione per trovare tutti i cicli del grafo
def dfs_cycle(graph, start_node, visited, parent, cycle, all_cycles):
    visited[start_node] = True
    cycle.append(start_node)

    # Explore neighbors
    for neighbor in graph.neighbors(start_node):
        if neighbor == parent:
            # Ignore the edge back to the parent node (avoid trivial cycles)
            continue
        if visited[neighbor]:
            # Found a cycle, extract it from the current path
            cycle_start_index = cycle.index(neighbor)
            new_cycle = cycle[cycle_start_index:]  # Get the cycle part
            all_cycles.append(new_cycle)
        else:
            # Recursively explore unvisited neighbors
            dfs_cycle(graph, neighbor, visited, start_node, cycle, all_cycles)

    # Backtrack: unmark the current node and remove it from the current cycle path
    visited[start_node] = False
    cycle.pop()

def find_all_cycles(graph):
    all_cycles = []
    visited = {node: False for node in graph.nodes()}

    # Start DFS from each node in the graph
    for node in graph.nodes():
        dfs_cycle(graph, node, visited, None, [], all_cycles)
    
    # Eliminate duplicates by sorting nodes within the cycle and using set
    seen = set()
    for cycle in all_cycles:
        cycle_sorted = tuple(sorted(cycle))
        if cycle_sorted not in seen:
            seen.add(cycle_sorted)
    
    return list(seen)

# Funzione che verifica la condizione dell'esercizio 3
def ex3(G, cycles):
    if len(cycles) != 2:
        return False
    
    n = G.number_of_nodes()
    # Ottieni gli archi dei due cicli
    cycle_edges = set()
    node_set = set()
    for cycle in cycles:
        if len(cycle) != n // 2:
            return False
        for node in cycle:
            # Verifica che non ci siano nodi in comune tra i due cicli
            if node in node_set:
                return False
            node_set.add(node)
        
        for i in range(len(cycle)):
            # Crea gli archi considerando che i cicli sono chiusi
            edge = (cycle[i], cycle[(i + 1) % len(cycle)])
            cycle_edges.add(tuple(sorted(edge)))  # Aggiungi l'arco come una coppia ordinata
    
    # Ottieni tutti gli archi nel grafo
    graph_edges = set(tuple(sorted(edge)) for edge in G.edges())
    
    # Verifica se gli archi nel grafo sono esattamente quelli dei cicli
    return graph_edges == cycle_edges 

# Funzione che verifica la condizione dell'esercizio 4
def ex4(G, cycles):
    if len(cycles) != 2:
        return False
    
    n = G.number_of_nodes()
    # Ottieni gli archi dei due cicli
    cycle_edges = set()
    node_set = set()
    for cycle in cycles:
        for node in cycle:
            # Verifica che non ci siano nodi in comune tra i due cicli
            if node in node_set:
                return False
            node_set.add(node)
        
        for i in range(len(cycle)):
            # Crea gli archi considerando che i cicli sono chiusi
            edge = (cycle[i], cycle[(i + 1) % len(cycle)])
            cycle_edges.add(tuple(sorted(edge)))  # Aggiungi l'arco come una coppia ordinata
    
    # Ottieni tutti gli archi nel grafo
    graph_edges = set(tuple(sorted(edge)) for edge in G.edges())
    
    # Verifica se gli archi nel grafo sono esattamente quelli dei cicli
    return graph_edges == cycle_edges

def ex5(G):
    # Calcola il grado medio del grafo
    total_degree = sum(dict(G.degree()).values())  # Somma di tutti i gradi dei nodi
    num_nodes = G.number_of_nodes()  # Numero totale di nodi nel grafo
    average_degree = total_degree / num_nodes if num_nodes > 0 else 0  # Gestione del caso di grafo vuoto
    return average_degree

def ex6(G):
    return len(G.edges())

def ex7(G):
    # Define the edges of the papillon structure
    papillon_edges = [
        (0, 1), (0, 2), (1, 2),  # Triangle part v1, v2, v3
        (0, 3), (0, 4), (3, 4)   # "Wings" formed by v1, v4, v5
    ]
    
    unique_papillons = set()  # Set to store unique papillon edge sets

    # Check all combinations of 5 nodes in the graph
    for nodes in itertools.combinations(G.nodes(), 5):
        subgraph = G.subgraph(nodes)  # Create a subgraph with the chosen 5 nodes

        # Try every permutation of the 5 nodes as a potential mapping to papillon roles
        for permuted_nodes in itertools.permutations(nodes):
            # Map permuted nodes to papillon node roles
            node_map = {i: permuted_nodes[i] for i in range(5)}
            
            # Map the edges of the *papillon* pattern to the nodes in the current permutation
            papillon_edges_mapped = {(min(node_map[u], node_map[v]), max(node_map[u], node_map[v])) for u, v in papillon_edges}
            
            # Check if the subgraph contains exactly the papillon edges
            if subgraph.size() == 6 and all(subgraph.has_edge(u, v) for u, v in papillon_edges_mapped):
                # Add the frozen set of edges as a unique papillon
                unique_papillons.add(frozenset(papillon_edges_mapped))

    return len(unique_papillons)  # Return the count of unique papillon subgraphs

# Parametri per il grafo Erdős-Rényi
n = 6 # Numero di nodi (deve essere pari per suddividere in cicli di n/2 nodi)
p = 0.3  # Probabilità di creazione degli archi
num_iterations = int(1e6)  # Numero di iterazioni (puoi ridurre per test più veloci)
successes = 0
EXERCISE = 7

# Esegui il test per num_iterations volte
for i in tqdm(range(num_iterations)):
    # Crea il grafo
    G = erdos_renyi_graph(n, p)

    # Trova tutti i cicli del grafo
    cycles = find_all_cycles(G)
    
    # Verifica la condizione dell'esercizio 3
    res = None
    if EXERCISE == 3:
        res = ex3(G, cycles)
    elif EXERCISE == 4:
        res = ex4(G, cycles)
    elif EXERCISE == 5:
        res = ex5(G)
    elif EXERCISE == 6:
        res = ex6(G)
    elif EXERCISE == 7:
        res = ex7(G)

    if EXERCISE != 5:
        successes += int(res)
    else:
        successes += res

    if res:
        print("PORCAMADONNA",res)
        plot_graph(G)


#check correctness
print(successes / num_iterations)

def multinomial_coeff(n, ks):
    assert(n == sum(ks))
    numerator = factorial(n)
    denominator = 1
    for k in ks:
        denominator *= factorial(k)
    return numerator // denominator


if EXERCISE == 3:
    assert(n % 2 == 0)
    res = multinomial_coeff(n,[n // 2]*2) * (p**n) * ((1-p)**(multinomial_coeff(n, [2,n-2])-n)) * (factorial(n//2-1)**2) / 8
elif EXERCISE == 4:
    res = 0.0
    for i in range(3, n-2):
        for j in range(3, n-i+1):
            res += multinomial_coeff(n,[i,j,n-i-j]) * factorial(i-1) * factorial(j-1) * p**(i+j) * (1-p)**(multinomial_coeff(n, [2,n-2])-i-j) / 8 
elif EXERCISE == 5:
    res = (n-1)*p
elif EXERCISE == 6:
    res = multinomial_coeff(n, [2,n-2])*p
elif EXERCISE == 7:
    res = multinomial_coeff(n, [5,n-5])*(p**6)*((1-p)**(4+(n-5)*5))
print(res)

