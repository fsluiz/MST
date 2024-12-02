import itertools
import networkx as nx
import matplotlib.pyplot as plt
from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import heapq
import random
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import heapq
from collections import defaultdict
###################################################################################
# Quantum Walk
def create_complete_graph(num_nodes):
    graph = nx.complete_graph(num_nodes)
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] = np.random.randint(1, 20)  # Pesos aleatórios
    return graph
# Função que calcula o número de qubits necessário para representar os nós
def qubits_number(num_nodes):
    if (np.log2(num_nodes)).is_integer():
        qubits_n = int(np.log2(num_nodes))
    else:
        qubits_n = int(np.log2(num_nodes)) + 1
    return qubits_n
#Função que constroi o Hamiltoniano baseado no Laplaciano
def construct_hamiltonian(graph):
    #Matrix adjacencia do grafo
    adj_matrix = nx.adjacency_matrix(graph, weight='weight').todense()
    #Matrix Degree do grafo
    degree = np.diag(np.sum(adj_matrix, axis=1).flatten())
    num_nodes = len(graph.nodes)
    # Calcula o número de qubits e define a dimensão da matriz
    qubits_n = qubits_number(num_nodes)
    dim = 2 ** qubits_n  # Dimensão da matriz para o número de qubits
    
    # Cria a matriz do Hamiltoniano com zeros
    H = np.zeros((dim, dim), dtype=complex)
    
    # Preenche o Hamiltoniano conforme as informações do grafo
    degree_non_zero = np.nonzero(degree)
    H[degree_non_zero] = -1 / degree[degree_non_zero]
    non_zero = np.nonzero(adj_matrix)
    H[non_zero] = -1 / adj_matrix[non_zero]
    
    # Define as dimensões do sistema para `Qobj`
    dims = [[2] * qubits_n, [2] * qubits_n]
    return Qobj(H, dims=dims)
# Cria o operador de evolução (evitando recriação desnecessária)
def quantum_walk(graph, t_max):
    hamiltonian = construct_hamiltonian(graph)
    return (-1j * hamiltonian * t_max).expm()
#Cria base de estados na base dos qubits
def qubit_basis_states(num_qubits):
    # Lista para armazenar todos os estados da base para o sistema de num_qubits
    basis_states = []
    
    # Itera sobre todos os estados de base binários de 0 até 2^num_qubits - 1
    for i in range(2 ** num_qubits):
        # Cria o estado binário correspondente e inicializa uma lista para o produto tensorial
        binary_state = f"{i:0{num_qubits}b}"  # Estado binário com padding até num_qubits
        state = [basis(2, int(bit)) for bit in binary_state]  # Estado correspondente
        
        # Produto tensorial para criar o estado de qubits
        basis_states.append(tensor(*state))
    
    return basis_states
# Função para calcular as probabilidades de transição entre estados

def prob_transition(graph, t_max, num_nodes):
    transition_probs = np.zeros((num_nodes, num_nodes))
    evolution_operator = quantum_walk(graph, t_max)
    num_qubits = qubits_number(num_nodes)
    # Pré-calcular as bases
    basis_vectors = qubit_basis_states(num_qubits)
    
    for i in range(num_nodes):
        ket_i = basis_vectors[i]
        for j in range(i, num_nodes):
            ket_j = basis_vectors[j]
            prob = np.abs((ket_j.dag() * evolution_operator * ket_i))**2
            transition_probs[i, j] = prob
            transition_probs[j, i] = prob  # Simetria
    return transition_probs

# Gera a MST com base nas probabilidades de transição
def quantum_walk_mst(graph, t_max, max_degree):
    num_nodes = len(graph.nodes)
    edge_probs = prob_transition(graph, t_max, num_nodes)
    edges_with_probs = []

    for u, v in graph.edges():
        prob = edge_probs[u, v]
        edges_with_probs.append((u, v, prob))

    # Ordena as arestas pela probabilidade de transição
    ordered_edges = sorted(edges_with_probs, key=lambda x: x[2], reverse=True)
    #print(ordered_edges)
    # MST usando NetworkX
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes)
    
       # Dicionário para controlar o grau dos nós na MST
    node_degrees = {node: 0 for node in graph.nodes}

    for u, v, _ in ordered_edges:
        # Verifica se a adição da aresta não viola a restrição de grau máximo
        if node_degrees[u] < max_degree and node_degrees[v] < max_degree:
            # Adiciona a aresta, se não houver ciclo e o grau máximo não for excedido
            if not nx.has_path(mst, u, v):  # Evita ciclos
                mst.add_edge(u, v, weight=graph.edges[u, v]['weight'])
                # Atualiza o grau dos nós
                node_degrees[u] += 1
                node_degrees[v] += 1
    mst_edges = list(mst.edges(data=True))
    return [(u, v, data['weight']) for u, v, data in mst_edges]
# Função para calcular o peso total das arestas na MST
def calculate_total_weight(mst_edges):
    return sum(weight for _, _, weight in mst_edges)
#####################################################################
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def kruskal_mdc(vertices, edges, max_degree):
    # Ordena as arestas pelo peso
    edges.sort(key=lambda x: x[2])  # x[2] é o peso da aresta

    uf = UnionFind(vertices)
    mst = []
    degree = [0] * vertices  # Inicializa o grau de cada vértice

    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):  # Verifica se u e v estão em componentes diferentes
            if degree[u] < max_degree and degree[v] < max_degree:
                uf.union(u, v)
                mst.append((u, v, weight))
                degree[u] += 1
                degree[v] += 1

    return mst
#Prim modificado
def prim_mdc(graph, max_degree):
    mst_edges = []
    total_weight = 0
    degree = defaultdict(int)
    num_nodes = len(graph.nodes)  # Ajuste conforme a representação do grafo
    uf = UnionFind(num_nodes)
        
    min_heap = []
    start_node = list(graph.nodes())[0]
    visited = set([start_node])
    
    for neighbor, weight in graph[start_node].items():
        heapq.heappush(min_heap, (weight['weight'], start_node, neighbor))
    
    while min_heap and len(visited) < len(graph.nodes()):
        weight, u, v = heapq.heappop(min_heap)
        
        if uf.find(u) != uf.find(v) and degree[u] < max_degree and degree[v] < max_degree:
            mst_edges.append((u, v, weight))
            total_weight += weight
            degree[u] += 1
            degree[v] += 1
            uf.union(u, v)
            visited.add(v)
            
            for neighbor, weight in graph[v].items():
                if neighbor not in visited:
                    heapq.heappush(min_heap, (weight['weight'], v, neighbor))
    
    return mst_edges, total_weight

# Algoritmo ganancioso
def greedy_approximation(graph, max_degree):
    mst_edges = []
    total_weight = 0
    degree = defaultdict(int)

    num_nodes = len(graph.nodes)  # Ajuste conforme a representação do grafo
    uf = UnionFind(num_nodes)
    
    
    min_heap = []
    for u, v in graph.edges():
        weight = graph[u][v]['weight']
        heapq.heappush(min_heap, (weight, u, v))
    
    while min_heap and len(mst_edges) < len(graph.nodes()) - 1:
        weight, u, v = heapq.heappop(min_heap)
        
        if uf.find(u) != uf.find(v) and degree[u] < max_degree and degree[v] < max_degree:
            mst_edges.append((u, v, weight))
            total_weight += weight
            degree[u] += 1
            degree[v] += 1
            uf.union(u, v)
    
    return mst_edges, total_weight

# Algoritmo de colonian de formigas

class AntColony:
    def __init__(self, graph, num_ants, max_degree, alpha=1, beta=1, evaporation_rate=0.5, iterations=100):
        self.graph = graph
        self.num_ants = num_ants
        self.max_degree = max_degree
        self.alpha = alpha  # Peso da feromona
        self.beta = beta    # Peso da heurística
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone = np.ones((len(graph.nodes()), len(graph.nodes())))  # Inicializa a matriz de feromona

    def heuristic(self, u, v):
        return 1 / self.graph[u][v]['weight']  # Heurística inversa do peso

    def update_pheromone(self, best_path, best_weight):
        for u, v in best_path:
            self.pheromone[u][v] += 1 / best_weight  # Atualiza a feromona

    def evaporate_pheromone(self):
        self.pheromone *= (1 - self.evaporation_rate)  # Evaporação da feromona

    def construct_solution(self):
        num_nodes = len(self.graph.nodes) 
        uf = UnionFind(num_nodes)
        path = []
        total_weight = 0
        degree = defaultdict(int)
        current_node = random.choice(list(self.graph.nodes()))
        visited = {current_node}
        
        
        while len(visited) < len(self.graph.nodes()):
            neighbors = [n for n in self.graph.neighbors(current_node) if n not in visited]
            if not neighbors:
                break

            probabilities = []
            for neighbor in neighbors:
                if degree[current_node] < self.max_degree and degree[neighbor] < self.max_degree and uf.find(current_node) != uf.find(neighbor):
                    pheromone_val = self.pheromone[current_node][neighbor] ** self.alpha
                    heuristic = (1 / self.graph[current_node][neighbor]['weight']) ** self.beta
                    probabilities.append((neighbor, pheromone_val * heuristic))
                else:
                    probabilities.append((neighbor, 0))
            
            total = sum(p[1] for p in probabilities)
            probabilities = [(p[0], p[1] / total) for p in probabilities if total > 0]
            
            next_node = np.random.choice([p[0] for p in probabilities], p=[p[1] for p in probabilities])
            path.append((current_node, next_node))
            total_weight += self.graph[current_node][next_node]['weight']
            visited.add(next_node)
            degree[current_node] += 1
            degree[next_node] += 1
            uf.union(current_node, next_node)
            current_node = next_node

        return path, total_weight


    def run(self):
        best_path = None
        best_weight = float('inf')

        for _ in range(self.iterations):
            for _ in range(self.num_ants):
                path, weight = self.construct_solution()
                if weight < best_weight:
                    best_weight = weight
                    best_path = path

            self.update_pheromone(best_path, best_weight)
            self.evaporate_pheromone()

        return best_path, best_weight
######################################################

# Função para processar uma combinação de nó e grau
def process_node_degree(node, degree, t_max):
    G = create_complete_graph(node)
    # Extrair arestas e pesos
    edges = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    
    # Aplicar os algoritmos
    mst_kruskal_mdc = kruskal_mdc(node, edges, degree)
    kruskal_mdc_weight = calculate_total_weight(mst_kruskal_mdc)
    
    num_ants = 60  # Número de formigas
    iterations = 100  # Número de iterações
    aco = AntColony(G, num_ants, degree, iterations=iterations)
    best_path, best_weight = aco.run()
    
    mst_prim_mod, total_weight_prim_mod = prim_mdc(G, degree)
    mst_greedy, total_weight_greedy = greedy_approximation(G, degree)
    quantum_mst_edges = quantum_walk_mst(G, t_max, degree)
    quantum_mst_weight = calculate_total_weight(quantum_mst_edges)
    
    # Retornar os resultados como um dicionário
    return {
        'num_nodes': node,
        'degree': degree,
        'edges': edges,
        'qw_weight': quantum_mst_weight,
        'kruskal_mdc_weight': kruskal_mdc_weight,
        'prim_mdc_weight': total_weight_prim_mod,
        'greed': total_weight_greedy,
        'antcolony': best_weight
    }
def unpack_task(task):
    return process_node_degree(*task)
# Configuração para paralelismo
if __name__ == "__main__":
    t_max = 0.1
    results = []
    node_range = range(4, 104)
    
    # Preparar lista de tarefas (nó, grau)
    tasks = [(node, degree, t_max) for node in node_range for degree in range(2, node)]
    
    # Barra de progresso
    with tqdm(total=len(tasks), desc="Processing") as pbar:
        # Pool de workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            # Executar as tarefas em paralelo
            for result in executor.map(unpack_task, tasks):
                results.append(result)
                pbar.update(1)  # Atualizar barra de progresso
    
    # Criar DataFrame e salvar como CSV
    df = pd.DataFrame(results)
    file_name = 'grafo_QW_others_4_104.csv'
    df.to_csv(file_name, index=False)
