import networkx as nx
import operator
import matplotlib.pyplot as plt
from networkx import spring_layout
from networkx import draw_networkx_edges

# --------------------------------------- CONFIGURABLE PARAMETERS --------------------------------------------------- #

# (integer) maximum number of iterations in case values do not converge normally
maximum_iterations = 150

# (float) damping parameter to make random jumps in case of dead-ends or spider-traps
beta = 0.45

# (boolean) flag to toggle between personalized and non-personalized page rank
personalized_page_rank = False  # if you make this false, set custom_category = 'none'

# (dictionary key: node value: personalization) used for personalized page rank
vector_set = {}

# (float) maximum error tolerated per node when checking for convergence
max_error = 0.000001

# data set file to be used
current_data_set_file = "data.txt"

# (string) category for personalized page rank
custom_category = 'none'
custom = {}
custom['students'] = {'node_size_multi': 100000, 'top': 10, 'threshold': 0.05}
custom['none'] = {'node_size_multi': 150000, 'top': 15, 'threshold': 0.01}
custom['docs'] = {'node_size_multi': 150000, 'top': 15, 'threshold': 0.009}

# -------------------------------------------- DATA-SET OPERATIONS -------------------------------------------------- #

categorized_web_pages = {}


# The create_categories_from_data_set() reads the data set and categorize the web pages into different sub-domains
# such as 'academics', 'admissions', etc.

# The data is stored in the dictionary 'categorized_web_pages' with category name as key and web-page ids as values.


def create_categories_from_data_set():
    with open('labels.txt') as textFile:
        for line in textFile:
            url = line.split(' ')
            domain = url[1][7:11]
            if domain == 'www1':
                rest = url[1][24:]
            else:
                rest = url[1][23:]
            category = rest[:rest.find('/')].lower()
            if category == '':
                continue
            if category in categorized_web_pages.keys():
                categorized_web_pages[category].append(url[0])
            else:
                categorized_web_pages.update({category: [url[0]]})


# -------------------------------------------- PAGE-RANK ALGORITHM -------------------------------------------------- #

# The calculate_page_rank() returns a dictionary the page rank of the nodes in the graph
# graph - a directed networkx graph created from the edge list of the network present in the data-set file


def calculate_page_rank(graph, weight='weight'):
    # ---------------------------------- CREATING INITIAL MATRICES -------------------------------------------------- #

    # Create a stochastic matrix where the column adds up to 1
    # Make a copy of the graph rather than modifying the original graph
    # Edge weight is 1 (non-weighted edges)
    stochastic = nx.stochastic_graph(graph, copy=True)  # Directed Graph
    no_of_nodes = stochastic.number_of_nodes()

    # ----------------------------- CONFIGURATION FOR PERSONALIZED PAGE RANK ---------------------------------------- #

    # starting page-rank for each node is 1 / no. of nodes
    rank_vector = dict.fromkeys(stochastic, 1.0 / no_of_nodes)

    if not personalized_page_rank:
        # Keep personalized rank values the same as rank vector values
        personalization_values = dict.fromkeys(stochastic, 1.0 / no_of_nodes)
    else:
        # Make personalized rank values equal to the stochastic values of pages in the mentioned category
        for i in range(1, 6013):
            vector_set.update({str(i): 0})
        docs = categorized_web_pages[custom_category]

        for ele in docs:
            vector_set[ele] = 1 / len(categorized_web_pages[custom_category])

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
        previous_rank_vector = rank_vector
        rank_vector = dict.fromkeys(previous_rank_vector.keys(), 0)
        dangle_sum = beta * sum(previous_rank_vector[n] for n in dangling_nodes)

        for n in rank_vector:
            # calculate the current matrix as a product of previous state matrix and stochastic matrix

            for j in stochastic[n]:
                rank_vector[j] += beta * previous_rank_vector[n] * stochastic[n][j][weight]

            # random surfing step to either jump or move forward
            rank_vector[n] += dangle_sum * dangling_weights[n] + (1.0 - beta) * personalization_values[n]

        # ---------------------------------- CHECK CONVERGENCE IN EACH ITERATION ------------------------------------ #

        total_diff = []
        for n in rank_vector:
            total_diff.append(abs(rank_vector[n] - previous_rank_vector[n]))
        total_error = sum(total_diff)

        if total_error < no_of_nodes * max_error:
            return rank_vector  # results converged

    # Max iterations crossed but results did not converge
    print('Failed to converge.')
    exit(0)


def main():
    with open(current_data_set_file) as textFile:

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
        to_plot = sorted_page_ranks[0:custom[custom_category]['top']]  # take top 10 page rank nodes
        for entry in to_plot:
            node = entry[0]
            for edge in edge_list:
                if edge[0] == node:
                    if pr.get(edge[1]) > custom[custom_category]['threshold']:  # threshold of neighbours
                        graph1.add_edge(edge[0], edge[1])

        color_map = []  # A list to store the colors for each node
        labels = {}
        for node1 in graph1.nodes():
            labels[node1] = node1
            if pr.get(node1) > 0.003:
                color_map.append('grey')  # Color top page rank nodes as blue
            else:
                color_map.append('green')  # Color rest of the nodes as green

        pos = spring_layout(graph1, k=0.9, iterations=5)
        nxnodes = nx.draw_networkx_nodes(graph1, pos, linewidths=2, node_color=color_map,
                                         node_size=[pr.get(v) * custom[custom_category]['node_size_multi']
                                                    for v in graph1.nodes()], label=True)
        nxnodes.set_edgecolor('black')
        nx.draw_networkx_labels(graph1, pos, labels, font_size=14)
        draw_networkx_edges(graph1, pos)
        plt.show()
        print('-' * 150)

# ---------------------------------------------------- DRIVER ------------------------------------------------------- #

create_categories_from_data_set()
main()
