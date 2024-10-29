import itertools
import networkx as nx

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

# Funzione per generare tutti i grafi con due cicli separati
def generate_separated_graphs(n):
    assert n % 2 == 0, "n deve essere pari"
    half = n // 2
    
    # Genera i nodi
    nodes = list(range(n))
    
    # Genera tutte le possibili suddivisioni in due sottografi disgiunti
    node_partitions = list(itertools.combinations(nodes, half))
    
    # Set per memorizzare tutti i grafi unici
    unique_graphs = set()
    
    for partition in node_partitions:
        # Ottieni i due insiemi di nodi
        set1 = list(partition)
        set2 = list(set(nodes) - set(set1))
        
        # Genera tutti i cicli per ciascun insieme di nodi
        cycles_set1 = generate_cycles(set1)
        cycles_set2 = generate_cycles(set2)
        
        #print(partition)
        for cycle1 in cycles_set1:
            for cycle2 in cycles_set2:
                # Crea grafi separati per ogni ciclo
                G1 = create_cycle_graph(cycle1)
                G2 = create_cycle_graph(cycle2)
                
                # Unisci i due grafi in un unico grafo
                G = nx.Graph()
                G.add_edges_from(G1.edges())
                G.add_edges_from(G2.edges())
                
                # Memorizza il grafo come un insieme ordinato di archi (tuple)
                edges_set = frozenset((min(u, v), max(u, v)) for u, v in G.edges())
                unique_graphs.add(edges_set)
        #return unique_graphs
    return unique_graphs

# Funzione principale per eseguire il calcolo
def main():
    n = 10  # Puoi cambiare n per grafi più grandi
    print(f"Generazione di grafi separati con {n} nodi e 2 cicli di lunghezza {n//2}...")
    
    # Genera tutti i grafi unici con due cicli separati
    unique_graphs = generate_separated_graphs(n)
    
    # Mostra il numero di grafi unici trovati
    print(f"Trovati {len(unique_graphs)} grafi unici.")
    
    # Stampa i grafi trovati (le liste di archi)
    for i, graph_edges in enumerate(unique_graphs):
        print(f"Grafo {i+1}: {sorted(graph_edges)}")

if __name__ == "__main__":
    main()
