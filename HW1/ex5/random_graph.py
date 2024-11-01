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
    res = []
    for cycle in all_cycles:
        cycle_sorted = tuple(sorted(cycle))
        if cycle_sorted not in seen:
            seen.add(cycle_sorted)
            res.append(cycle)
    
    return res

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
    
    # Verifica che tutti i nodi siano in uno dei cicli
    if NEW_FOUR and len(node_set) != n:
        return False
    
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
    # Definisce gli archi della struttura del papillon
    papillon_edges = [
        (0, 1), (0, 2), (1, 2),  # Parte triangolare v1, v2, v3
        (0, 3), (0, 4), (3, 4)   # "Ali" formate da v1, v4, v5
    ]
    
    unique_papillons = 0  # Contatore per papillon unici

    # Definisci solo le 15 permutazioni distintive che mantengono le caratteristiche del papillon
    unique_permutations = [
        (0, 1, 2, 3, 4), (1, 0, 2, 3, 4), (2, 0, 1, 3, 4), (3, 0, 1, 2, 4), (4, 0, 1, 2, 3),
        (0, 1, 3, 2, 4), (1, 0, 3, 2, 4), (2, 0, 3, 1, 4), (3, 0, 2, 1, 4), (4, 0, 2, 1, 3),
        (0, 1, 4, 3, 2), (1, 0, 4, 3, 2), (2, 0, 4, 3, 1), (3, 0, 4, 2, 1), (4, 0, 3, 2, 1)
    ]

    # Controlla tutte le combinazioni di 5 nodi nel grafo
    for nodes in itertools.combinations(G.nodes(), 5):
        subgraph = G.subgraph(nodes)  # Crea un sottografo con i 5 nodi scelti

        # Prova solo le 15 permutazioni uniche
        for perm in unique_permutations:
            # Mappa i nodi permutati ai ruoli del papillon
            node_map = {i: nodes[perm[i]] for i in range(5)}
            
            # Mappa gli archi della struttura del papillon ai nodi della permutazione corrente
            papillon_edges_mapped = [(node_map[u], node_map[v]) for u, v in papillon_edges]
            
            # Controlla se il sottografo contiene esattamente gli archi del papillon
            if subgraph.size() == 6 and all(subgraph.has_edge(u, v) for u, v in papillon_edges_mapped):
                # Incrementa il contatore dei papillon unici trovati
                unique_papillons += 1
                break  # Interrompi il ciclo se trovi un papillon per questo insieme di nodi
                
    return unique_papillons  # Restituisci il conteggio dei papillon unici

def generate_unique_graph_permutations(G):
    unique_graphs = set()
    nodes = list(G.nodes())
    
    # Genera tutte le permutazioni dei nodi
    for perm in itertools.permutations(nodes):
        # Crea una mappatura dei nodi secondo la permutazione corrente
        mapping = {original: permuted for original, permuted in zip(nodes, perm)}
        # Applica la mappatura ai nodi del grafo per creare una nuova configurazione
        permuted_graph = nx.relabel_nodes(G, mapping, copy=True)
        
        # Genera una rappresentazione immutabile (frozenset) per controllare l'unicità del grafo
        permuted_edges = frozenset((min(u, v), max(u, v)) for u, v in permuted_graph.edges())
        
        # Aggiungi il grafo solo se non è già stato generato
        if permuted_edges not in unique_graphs:
            unique_graphs.add(permuted_edges)
            #print(f"Nuova configurazione trovata con permutazione: {perm}")
    
    # Restituisce il numero di grafi distinti generati
    return len(unique_graphs)


# Parametri per il grafo Erdős-Rényi
n = 8 # Numero di nodi (deve essere pari per suddividere in cicli di n/2 nodi)
p = 0.3  # Probabilità di creazione degli archi
num_iterations = int(1e7)  # Numero di iterazioni (puoi ridurre per test più veloci)
successes = 0
EXERCISE = 3
NEW_FOUR = False

print("n =",n)
print("tested exercise:",EXERCISE)

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

    #if res:
    #    plot_graph(G)


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
            res += multinomial_coeff(n,[i,j,n-i-j]) * factorial(i-1) * factorial(j-1) * (p**(i+j)) * ((1-p)**(multinomial_coeff(n, [2,n-2])-i-j)) / 8 
    if NEW_FOUR: 
        res = 0.0
        for i in range(3, n-2):
            res += multinomial_coeff(n, [i, n-i]) * factorial(i-1) * factorial(n-i-1) * (p**n) * ((1-p)**(multinomial_coeff(n,[2,n-2])-n)) / 8
   
elif EXERCISE == 5:
    res = (n-1)*p
elif EXERCISE == 6:
    res = multinomial_coeff(n, [2,n-2])*p
elif EXERCISE == 7:
    res = multinomial_coeff(n, [5,n-5])*(p**6)*((1-p)**4)*15
print(res)

