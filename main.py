import numpy as np


def ACO(graph, ants, iteration, alpha, beta, delta_tau, rho):

    opt_dist = None
    opt_path = None 
    pheromones = None

    inv_distances = inv_dis(graph)

    for i in range(iteration):
        dist = 0

        pos = np.random.randint(graph.shape[0], size = ants)

        paths = update_ants(graph, pos, inv_distances, pheromones, alpha, beta, delta_tau)

        for path in paths:

            for node in range(1, path.shape[0]):

                dist += np.sqrt(((graph[int(path[node])] - graph[int(path[node - 1])])**2).sum()) 

            if not opt_dist or dist < min_distance:
                min_distance = dist
                opt_path = path

        opt_path = np.append(opt_path, opt_path[0])
    
    return (opt_path, opt_dist)


def inv_dis(graph):
    dis = np.zeros((graph.shape[0], graph.shape[1]))

    for i, node in enumerate(graph):
        dis[i] = np.sqrt(((graph - node) ** 2 ).sum(axis=1))

    inverse = 1 / dis

    inverse[inverse == np.inf] = 0

    return inverse

def update_ants(graph, pos, inv_dist, pheromones, alpha, beta, delta_tau):
    
    paths = np.zeros((graph.shape[0], pos.shape[0]), dtype = int) - 1

    paths[0] = pos

    for node in range(1, graph.shape[0]):

        for ant in range(pos.shape[0]):

            next_loc_prob = (inv_dist[pos[ant]] ** alpha + pheromones[pos[ant]] ** beta /
                                            inv_dist[pos[ant]].sum() ** alpha + pheromones[pos[ant]].sum() ** beta)


            next_position = np.argwhere(next_loc_prob == np.amax(next_loc_prob))[0][0]


            while next_position in paths[:, ant]:

                next_loc_prob[next_position] = 0.0


                next_position = np.argwhere(next_loc_prob == np.amax(next_loc_prob))[0][0]


            paths[node, ant] = next_position


            pheromones[node, next_position] = pheromones[node, next_position] + delta_tau


    return np.swapaxes(paths, 0, 1)