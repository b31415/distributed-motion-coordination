"""
Simulation for thesis.
Recreate the simulations in section 4 of [1]. I.e. figures 3, 5, 7, 9, 11 and 12.

Author: Mark Amos
Date: 4/03/2022
References:
    [1] W. Ren and E. Atkins, "Distributed multi-vehicle coordinated control 
        via local information exchange", International Journal of Robust and 
        Nonlinear Control, vol. 17, no. 10-11, pp. 1002-1033, 2007. 
        Available: 10.1002/rnc.1147 [Accessed 4 March 2022].
        
TODO: Change graph_ids to enums
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

GRAPH_ID = 0
GAMMA = 1
DYNAMICS_TIME_STEP  = 1e-4
SIMULATION_TIME     = 10
SAVE_FIG = True
SAVE_FORMATS = ['pdf']
TITLES = {
    0: "Graph with Separate Components",
    1: "Graph with Multiple Leaders",
    2: "Connected Undirected Graph",
    3: "Leader-follower Graph",           # I.e. is a spanning tree
    4: "Graph containing Spanning Tree",  # One more edge than 3
    5: "Balanced Graph"                   # One more edge than 4
}

    
def create_graph(graph_id: int = 4, n_nodes = 4, states: list[dict] = None) -> nx.Graph:
    """
    Create a graph with one of the topologies in [1].
    
    Args:
        graph_id (int): An index referring to which graph to create out of:
            Topologies that require 4 nodes:
                0: Graph with 2 separate components (fig 2 in [1])
                1: Graph with 2 leaders (fig 4 in [1])
                2: Connected undirected graph (fig 6 in [1])
                3: Leader-follower graph (fig 8 in [1])
                4: Graph with a cycle and contains a spanning tree
                    (fig 10 in [1] which achieves consensus)
                5: Balanced graph (average consensus)
                6: Reverse of graph 4
            Topologies that support any number of nodes:
                7: Complete graph (TODO: doesn't support states != None)
                8: One leader (node 0), N - 1 followers, giving a tree of height 1
        n_nodes (int): Number of nodes in the graph. 
        states: Each element represents the initial states for a single node.
            The order of the list determines which node the values are assigned to.
                
    """
    if graph_id <= 6 and n_nodes != 4:
        raise ValueError("Graph ID {} requires 4 nodes, but got {} nodes".format(graph_id, n_nodes))
    
    if graph_id == 7 and states is not None:
        raise ValueError("Unsupported argument combination")
    
    def init_digraph_nodes():
        """Create digraph with nodes and their initial states, but no edges."""
        Gd = nx.DiGraph()
        if states:
            nodes = [(i, states[i]) for i in range(n_nodes)]
        else:
            nodes = list(range(n_nodes))
        Gd.add_nodes_from(nodes)
        return Gd
    
    # Topologies with 4 nodes
    def create_graph0():
        Gd = init_digraph_nodes()
        Gd.add_edges_from([(1, 0), (2, 3)])
        return Gd
    
    def create_graph1():
        Gd = create_graph0()
        Gd.add_edge(2, 1)
        return Gd
    
    def create_graph2():
        Gd = create_graph1()
        Gu = Gd.to_undirected()
        return Gu
    
    def create_graph3():
        Gd = init_digraph_nodes()
        Gd.add_edges_from([(1, 0), (3, 0), (2, 1)])
        return Gd
    
    def create_graph4():
        Gd = create_graph3()
        Gd.add_edge(0, 2)
        return Gd
    
    def create_graph5():
        Gd = create_graph4()
        Gd.add_edge(0, 3)
        return Gd
    
    def create_graph6():
        Gd = create_graph4()
        return Gd.reverse()

    # Topologies with N nodes
    def create_graph7():
        Gd = nx.complete_graph(n_nodes, nx.Digraph())
        return Gd
    
    def create_graph8():
        Gd = init_digraph_nodes()
        Gd.add_edges_from([(0, node) for node in range(1, n_nodes)])
        return Gd
    
    id_to_graph = [
        create_graph0,
        create_graph1,
        create_graph2,
        create_graph3,
        create_graph4,
        create_graph5,
        create_graph6,
        create_graph7,
        create_graph8
    ]
    
    return id_to_graph[graph_id]()
    
def pos(node_id):
    """
    Get the position (xi) of a node_id (i).
    Args:
        node_id (int)
    Returns:
        (float)
    """
    return positions[-1][node_id]

def velocity(node_id):
    """
    Get the velocity (zeta) of node_id (i)
    Args:
        node_id (int)
    Returns:
        (float)
    """
    return velocities[-1][node_id]

def u(node_id):
    """
    Compute ui = -sum((pos_i - pos_j) + gamma(velocity_i - velocity_j)) 
    over neighbours (vj) of vi. I.e. protocol 4 from [1] with kij = 1.
    
    Args:
        node_id (int): Index 'i' of the node to computer ui for.
    Return:
        (float) Result of the protocol
    """
    neighbours = list(G[node_id])  
    ui = sum([pos(j) + GAMMA*velocity(j) for j in neighbours]) \
        - len(neighbours) * (pos(node_id) + GAMMA*velocity(node_id))
    return ui

def run_simulation():
    for _ in times:
        # Approximate Equation 3 in [1] using a forward difference 
        new_positions = [pos(i) + DYNAMICS_TIME_STEP*velocity(i) for i in V]
        new_velocities = [velocity(i) + DYNAMICS_TIME_STEP*u(i) for i in V]
        
        # Set the new values for pos and velocity after all the calculations
        # for the current time-step of Equation 3 are done.
        positions.append(new_positions)
        velocities.append(new_velocities)

def plot():
    """Plot the positions and velocities"""
    # Setup
    ax: tuple[plt.Axes, plt.Axes]
    fig = plt.Figure
    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    plt.suptitle(f"Evolution of States for {TITLES[GRAPH_ID]} ($\gamma = {GAMMA}$)", 
        fontsize='x-large')

    # Plot positions
    plt.subplot(211)
    plt.ylabel(r"Positions, $\xi_i$ (m)", labelpad=-5, fontsize='large')
    plt.grid(True, linewidth=0.15)
    lines = plt.plot(times, positions[:-1], linewidth=1)
    plt.xlim([0, SIMULATION_TIME])

    # Plot velocities
    plt.subplot(212)
    plt.xlabel("Time (s)", labelpad=-5, fontsize='large')
    plt.ylabel(r"Velocities, $\zeta_i$ (m/s)", labelpad=-5, fontsize='large')
    plt.grid(True, linewidth=0.15)
    plt.plot(times, velocities[:-1], linewidth=1)
    plt.xlim([0, SIMULATION_TIME])
    fig.align_labels(ax)
    
    # Plot legend
    ax[1].legend(
        labels          = ['$A_1$', '$A_2$', '$A_3$', '$A_4$'], 
        loc             = 'lower center', 
        borderpad       = 1,
        ncol            = 4,
        title           = "Nodes",
        title_fontsize  = 'large',
        bbox_to_anchor  = (0.5, -0.45)
    )
    
    plt.tight_layout(h_pad=0.2, rect=(-0.01, -0.01, 1.01, 1))
    # Save and show plot
    if SAVE_FIG:
        for format in SAVE_FORMATS:
            os.makedirs(format, exist_ok=True)
            plt.savefig(f"{format}/recreate_2nd_order_fig{GRAPH_ID}.{format}", format=format)
    plt.show()

if __name__ == "__main__":
    """Create graph"""
    positions = [list(0.2*np.arange(4))]
    velocities = [list(0.1*np.arange(4))]

    G = create_graph(GRAPH_ID)
    V = list(G.nodes)

    """Simulate"""
    times = np.arange(0, SIMULATION_TIME, DYNAMICS_TIME_STEP)
    run_simulation()
    plot()
