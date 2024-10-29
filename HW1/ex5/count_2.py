import itertools
import networkx as nx
from collections import defaultdict

# Funzione per generare tutti i cicli di lunghezza k su un insieme di nodi
def generate_cycles(nodes):
    # Generate all possible permutations (cycles)
    return list(itertools.permutations(nodes, len(nodes)))

# Funzione per creare un ciclo chiuso da una sequenza di nodi
def create_cycle_graph(cycle):
    G = nx.Graph()
    n = len(cycle)
    for i in range(n):
        # Aggiungi un arco tra nodi consecutivi e chiudi il ciclo
        edge = (min(cycle[i], cycle[(i + 1) % n]), max(cycle[i], cycle[(i + 1) % n]))
        G.add_edge(*edge)
    return G

# Funzione per generare tutte le partizioni possibili di nodi per formare due cicli
def generate_valid_partitions(n):
    partitions = []
    # Trova tutte le possibili coppie di lunghezze di cicli
    for k in range(3, n - 2):  # k è la lunghezza del primo ciclo, deve essere almeno 3
        for j in range(3, n-2):  # Assicurati che entrambi i cicli abbiano almeno 3 nodi
            if j + k <= n:
                partitions.append((k, j))
    print(partitions)
    return partitions

# Funzione per generare tutti i grafi con due cicli separati e nodi disgiunti
def generate_separated_graphs(n):
    # Dizionario per contare i grafi unici in base al numero di archi
    edge_count_graphs = defaultdict(int)
    
    # Genera i nodi
    nodes = list(range(n))
    
    # Ottieni tutte le partizioni valide dei nodi per i cicli
    cycle_partitions = generate_valid_partitions(n)
    unique_graphs = set()
    
    for cycle_sizes in cycle_partitions:
        k, j = cycle_sizes
        # Ottieni tutte le possibili suddivisioni dei nodi per i due cicli
        for partition in itertools.combinations(nodes, k):
            set1 = list(partition)
            remaining_nodes = list(set(nodes) - set(set1))
                        
            # Ottieni tutte le possibili suddivisioni per il secondo ciclo
            for sub_partition in itertools.combinations(remaining_nodes, j):
                set2 = list(sub_partition)
                isolated_nodes = list(set(remaining_nodes) - set(set2))  # Nodi isolati rimasti

                # Genera tutti i cicli per ciascun insieme di nodi
                cycles_set1 = generate_cycles(set1)
                cycles_set2 = generate_cycles(set2)
                
                for cycle1 in cycles_set1:
                    for cycle2 in cycles_set2:
                        # Crea grafi separati per ogni ciclo
                        G1 = create_cycle_graph(cycle1)
                        G2 = create_cycle_graph(cycle2)
                        
                        # Unisci i due cicli e aggiungi nodi isolati
                        G = nx.Graph()
                        G.add_edges_from(G1.edges())
                        G.add_edges_from(G2.edges())
                        G.add_nodes_from(isolated_nodes)  # Aggiungi nodi isolati senza archi
                        
                        # Conta gli archi e salva il grafo come unico
                        edges_set = frozenset((min(u, v), max(u, v)) for u, v in G.edges())
                        if edges_set not in unique_graphs:
                            unique_graphs.add(edges_set)
                            edge_count_graphs[len(edges_set)] += 1
                            
    return edge_count_graphs

# Funzione principale per eseguire il calcolo
def main():
    n = 10  # Puoi cambiare n per grafi più grandi
    print(f"Generazione di grafi separati con {n} nodi e 2 cicli di diverse lunghezze...")
    
    # Genera tutti i grafi unici con due cicli separati e conta in base al numero di archi
    edge_count_graphs = generate_separated_graphs(n)
    
    # Mostra il numero di grafi unici trovati per ciascun numero di archi
    for edge_count, count in edge_count_graphs.items():
        print(f"Grafi con {edge_count} archi: {count} grafi unici.")
    print("total:", sum([edge_count_graphs[i] for i in edge_count_graphs]))
if __name__ == "__main__":
    main()
