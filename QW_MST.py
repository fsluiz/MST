# Utilizando a base binaria combinação de qubits
import networkx as nx
import numpy as np
from qutip import *
import matplotlib.pyplot as plt


# Função que cria grafos completos com pesos aleatórios
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
    H[degree_non_zero] = 1 / degree[degree_non_zero]
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
def quantum_walk_mst(graph, t_max):
    num_nodes = len(graph.nodes)
    edge_probs = prob_transition(graph, t_max, num_nodes)
    edges_with_probs = []

    for u, v in graph.edges():
        prob = edge_probs[u, v]
        edges_with_probs.append((u, v, prob))

    # Ordena as arestas pela probabilidade de transição
    ordered_edges = sorted(edges_with_probs, key=lambda x: x[2], reverse=True)
    
    # MST usando NetworkX
    mst = nx.Graph()
    mst.add_nodes_from(graph.nodes)
    
    for u, v, _ in ordered_edges:
        if not nx.has_path(mst, u, v):  # Evita ciclos
            mst.add_edge(u, v, weight=graph.edges[u, v]['weight'])

    mst_edges = list(mst.edges(data=True))
    return [(u, v, data['weight']) for u, v, data in mst_edges]

# Algoritmo de Kruskal para calcular a MST
def kruskal_mst(graph):
    mst_edges = list(nx.minimum_spanning_edges(graph, data=True))
    return [(u, v, data['weight']) for u, v, data in mst_edges]

# Função para plotar a MST
def plot_mst(graph, mst_edges, title):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    plt.title(title)

    # Desenhar o grafo original
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, edge_color='gray')

    # Adicionar os pesos de todas as arestas do grafo original
    all_edge_labels = {(u, v): f'{d["weight"]}' for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=all_edge_labels, font_color='blue')

    # Desenhar a MST em destaque
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(mst_edges)
    nx.draw(mst_graph, pos, with_labels=True, node_color='lightgreen', node_size=700, font_size=10, edge_color='red', width=2)
    
    # Adicionar pesos das arestas da MST no gráfico
    mst_edge_labels = {(u, v): f'{d}' for u, v, d in mst_edges}
    nx.draw_networkx_edge_labels(mst_graph, pos, edge_labels=mst_edge_labels, font_color='red')

    plt.show()

# Função para calcular as probabilidades de transição ao longo do tempo
def calculate_probabilities_over_time(graph, times, num_nodes):
    probs_over_time = []

    for t in times:
        probs = prob_transition(graph, t, num_nodes)
        probs_over_time.append(probs)

    return np.array(probs_over_time)

# Função para plotar as probabilidades ao longo do tempo
def plot_probabilities_over_time(times, probs_over_time, num_nodes):
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                plt.plot(times, probs_over_time[:, i, j], label=f'P({i} -> {j})')

    plt.xlabel('Tempo (t)')
    plt.ylabel('Probabilidade de Transição')
    plt.title('Variação das Probabilidades de Transição ao longo do Tempo')
    plt.legend()
    plt.show()

# Função para calcular o peso total das arestas na MST
def calculate_total_weight(mst_edges):
    return sum(weight for _, _, weight in mst_edges)
# Exemplo de uso
num_nodes = 5  # Definir o número de nós no grafo
graph = create_complete_graph(num_nodes)
t_max = 0.001 # Número de passos da caminhada quântica

# Construir a MST a partir da caminhada quântica
quantum_mst_edges = quantum_walk_mst(graph, t_max)
kruskal_mst_edges = kruskal_mst(graph)

# Calcular a soma dos pesos das MSTs
quantum_mst_weight = calculate_total_weight(quantum_mst_edges)
kruskal_mst_weight = calculate_total_weight(kruskal_mst_edges)

# Plotar as MSTs com pesos
plot_mst(graph, quantum_mst_edges, f"Quantum Walk MST (Peso total: {quantum_mst_weight})")
plot_mst(graph, kruskal_mst_edges, f"Kruskal MST (Peso total: {kruskal_mst_weight})")


# Comparar os pesos
print(f"Peso total da MST (Kruskal): {kruskal_mst_weight}")
print(f"Peso total da MST (Caminhada Quântica): {quantum_mst_weight}")
