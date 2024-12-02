import itertools
import networkx as nx
import matplotlib.pyplot as plt
from qutip import *
import matplotlib.pyplot as plt
import numpy as np

def is_valid_tree(graph, edges, max_degree):
    """Verifica se um subconjunto de arestas forma uma árvore válida com restrição de grau."""
    # Cria subgrafo com as arestas escolhidas
    subgraph = nx.Graph()
    subgraph.add_edges_from(edges)
    
    # Verifica se o subgrafo é uma árvore (conexo e acíclico)
    if not nx.is_tree(subgraph):
        return False
    
    # Verifica restrição de grau
    degrees = subgraph.degree()
    if any(degree > max_degree for node, degree in degrees):
        return False
    
    return True

def exhaustive_mst_with_degree_constraint(graph, max_degree):
    """Realiza busca exaustiva para encontrar a MST com restrição de grau máximo."""
    nodes = list(graph.nodes)
    edges = list(graph.edges(data=True))  # Inclui o peso das arestas
    min_weight = float('inf')
    best_tree = None

    # Gera todas as combinações de n-1 arestas (possíveis árvores geradoras)
    for edge_subset in itertools.combinations(edges, len(nodes) - 1):
        # Verifica se o subconjunto de arestas forma uma árvore válida
        if is_valid_tree(graph, edge_subset, max_degree):
            # Calcula o peso total do subgrafo
            total_weight = sum(edge[2]['weight'] for edge in edge_subset)
            # Atualiza a melhor árvore se o peso total for menor
            if total_weight < min_weight:
                min_weight = total_weight
                best_tree = edge_subset

    return best_tree, min_weight

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
def construct_hamiltonian(graph,max_degree):
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
    H[degree_non_zero] =1 / degree[degree_non_zero] -1/((degree[degree_non_zero])-max_degree)
    non_zero = np.nonzero(adj_matrix)
    H[non_zero] = -1 / adj_matrix[non_zero]
    
    # Define as dimensões do sistema para `Qobj`
    dims = [[2] * qubits_n, [2] * qubits_n]
    return Qobj(H, dims=dims)
# Cria o operador de evolução (evitando recriação desnecessária)
def quantum_walk(graph, t_max):
    hamiltonian = construct_hamiltonian(graph,max_degree)
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
    evolution_operator = quantum_walk(graph, t_max)
    num_qubits = qubits_number(num_nodes)
    basis_vectors = qubit_basis_states(num_qubits)
    
    # Constrói uma matriz onde cada coluna é um estado da base
    basis_matrix = np.column_stack([basis.full() for basis in basis_vectors])
    
    # Aplica a evolução a todos os estados de uma vez
    evolved_states = evolution_operator.full() @ basis_matrix
    
    # Calcula as probabilidades de transição
    transition_probs = np.abs(np.dot(basis_matrix.T.conj(), evolved_states))**2
    
    return transition_probs

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

    mst_edges = list(mst.edges(data=True))
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
# Função para plotar lado a lado o grafo completo com MST e caminhada quântica
def plot_graph_and_paths(graph, mst_edges, mst_weight, prob_edges, prob_weight):
    pos = nx.spring_layout(graph)

    # Configuração para exibir dois gráficos lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Primeiro gráfico: Grafo completo e caminho MST
    nx.draw(graph, pos, with_labels=True, ax=axes[0], node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _ in mst_edges], edge_color="red", width=2, ax=axes[0])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): d['weight'] for u, v, d in graph.edges(data=True)}, ax=axes[0])
    axes[0].set_title(f"MST com Restrição de Grau (Peso Total: {mst_weight})")
    
    # Segundo gráfico: Grafo completo e caminho da caminhada quântica
    nx.draw(graph, pos, with_labels=True, ax=axes[1], node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _ in prob_edges], edge_color="blue", width=2, ax=axes[1])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): d['weight'] for u, v, d in graph.edges(data=True)}, ax=axes[1])
    axes[1].set_title(f"Caminhada Quântica - Caminho de Máx. Prob. (Peso Total: {prob_weight})")
    
    plt.show()
# Função para calcular o peso total das arestas na MST
def calculate_total_weight(mst_edges):
    return sum(weight for _, _, weight in mst_edges)

# Exemplo de uso
# Criando um grafo de teste
num_nodes = 5
G = create_complete_graph(num_nodes)

# Parâmetro de grau máximo
max_degree = 2
t_max = 0.1
# Encontrando a MST com restrição de grau
mst_edges, mst_weight = exhaustive_mst_with_degree_constraint(G, max_degree)
# Construir a MST a partir da caminhada quântica
quantum_mst_edges = quantum_walk_mst(G, t_max, max_degree)

# Calcular a soma dos pesos das MSTs
quantum_mst_weight = calculate_total_weight(quantum_mst_edges)

# Plotando o grafo e a árvore geradora mínima
plot_graph_and_paths(G, mst_edges, mst_weight, quantum_mst_edges, quantum_mst_weight)
