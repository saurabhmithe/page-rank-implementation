import networkx as nx
import operator
import matplotlib.pyplot as plt

# --------------------------------------- CONFIGURABLE PARAMETERS -------------------------------------------------- #

maximum_iterations = 150  # (integer) maximum number of iterations in case values do not converge normally
beta = 0.85  # (float) damping parameter to make random jumps in case of dead-ends or spider-traps
personalized_page_rank = False  # (boolean) flag to toggle between personalized and non-personalized page rank
vector_set = {}  # (dictionary k: node v: personalization) used for personalized page rank
max_error = 0.000001  # (float) maximum error tolerated when checking for convergence

current_dataset_file = "wiki-Vote.txt"

# Returns a dictionary the page rank of the nodes in the graph
# graph - a directed networkx graph created from the edge list of the network present in the data-set file


def calculate_page_rank(graph, weight='weight'):

    # ---------------------------------- CREATING INITIAL MATRICES -------------------------------------------------- #

    # Create a stochastic matrix where the column adds up to 1
    # Make a copy of the graph rather than modifying the original graph
    # Edge weight is 1 (non-weighted edges)
    stochastic = nx.stochastic_graph(graph, copy=True)
    no_of_nodes = stochastic.number_of_nodes()

    # ----------------------------- CONFIGURATION FOR PERSONALIZED PAGE RANK ---------------------------------------- #

    # starting page-rank for each node is 1
    current_matrix = dict.fromkeys(stochastic, 1.0 / no_of_nodes)

    if not personalized_page_rank:
        # Assign uniform personalization vector
        personalization_values = dict.fromkeys(stochastic, 1.0 / no_of_nodes)
    else:
        if set(graph) - set(vector_set):
            print('Personalized set not structured completely. Exiting.')
            exit(0)

        s = float(sum(vector_set.values()))
        personalization_values = dict((k, v / s) for k, v in vector_set.items())

    # Use personalization vector if dangling vector not specified
    dangling_weights = personalization_values
    dangling_nodes = [n for n in stochastic if stochastic.out_degree(n, None) == 0.0]

    # ----------------------------------------- RUN ITERATIONS ------------------------------------------------------ #

    # start iterations to calculate page ranks
    for iteration in range(maximum_iterations):
        previous_matrix = current_matrix
        current_matrix = dict.fromkeys(previous_matrix.keys(), 0)

        danglesum = beta * sum(previous_matrix[n] for n in dangling_nodes)

        for n in current_matrix:
            # calculate the current matrix as a product of previous state matrix and stochastic matrix

            for nbr in stochastic[n]:
                current_matrix[nbr] += beta * previous_matrix[n] * stochastic[n][nbr][weight]

            # random surfing step
            current_matrix[n] += danglesum * dangling_weights[n] + (1.0 - beta) * personalization_values[n]

        # ---------------------------------- CHECK CONVERGENCE IN EACH ITERATION ------------------------------------ #

        total_diff = []
        for n in current_matrix:
            total_diff.append(abs(current_matrix[n] - previous_matrix[n]))
            total_error = sum(total_diff)

        if total_error < no_of_nodes * max_error:
            return current_matrix  # results converged

    # Max iterations crossed but results did not converge
    print('Failed to converge.')
    exit(0)


def main():
    with open(current_dataset_file) as textFile:

        # ---------------------------------- CREATE GRAPH FROM EDGE LIST -------------------------------------------- #

        edge_list = [line.split() for line in textFile]
        print('Finished reading the edge-list from the data-set file')
        print('-' * 150)

        graph = nx.DiGraph()  # create an empty directed graph
        graph.add_edges_from(edge_list)  # populate the graph with edges from the edge list
        print('Created a networkx graph form the edge-list')
        print('-' * 150)

        # ------------------------------------------ CALCULATE PAGE RANK -------------------------------------------- #

        pr = calculate_page_rank(graph)  # calculate page rank for the graph
        print('Finished calculating page rank for each node')
        print('-' * 150)

        # print the page rank of each node
        print('Printing all calculated page ranks from highest to lowest (up to 4 precision level')
        sorted_page_ranks = sorted(pr.items(), key=operator.itemgetter(1), reverse=True)
        for entry in sorted_page_ranks:
            print(entry[0] + ": " + str(round(entry[1], 4)))
        print('-' * 150)

        # ----------------------------------------- PLOT HIGH RANK NODES -------------------------------------------- #

        # Create the sub-graph to be plotted
        print('Plotting the graph')
        graph1 = nx.DiGraph()
        to_plot = sorted_page_ranks[0:10]  # take top 10 page rank nodes
        for entry in to_plot:
            node = entry[0]
            for edge in edge_list:
                if edge[0] == node:
                    if pr.get(edge[1]) > 0.0010:  # threshold of neighbours
                        graph1.add_edge(edge[0], edge[1])

        color_map = []  # A list to store the colors for each node
        for node1 in graph1.nodes():
            if pr.get(node1) > 0.0015:
                color_map.append('blue')  # Color top page rank nodes as blue
            else:
                color_map.append('green')  # Color rest of the nodes as green

        nx.draw(graph1, node_color=color_map, node_size=[pr.get(v) * 500000 for v in graph1.nodes()], with_labels=True)
        plt.show()
        print('-' * 150)


main()
