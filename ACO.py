import math
import numpy as np
from sklearn.preprocessing import normalize
import random

#calculo de distancia euclideana entre dos puntos del grafo
def dist(city1,city2):
    return math.sqrt((city1[1] - city2[1])**2 + (city1[2]-city2[2])**2)
    
    
def ACO(graph, ants, iterations, alpha, beta, rho, Q):

    dist_opt = float("inf")
    camino_opt = []

    #inicializacion de feromonas
    pheromones = np.array([[1.0 for j in range(len(graph))] for i in range(len(graph))])

    #posiciones iniciales hormigas
    positions = [random.randint(0,len(graph)-1) for i in range(ants)]
    
    #calculo distancias inversas
    inv_dist = [[0.0 for i in range(len(graph))] for j in range(len(graph))]
    for i in range(len(graph)):
        for j in range(len(graph)):
            if i==j:
                inv_dist[i][j] = 0.0
            else:
                inv_dist[i][j] = 1 / dist(graph[i],graph[j])

    #iteración del algoritmo
    for i in range(iterations):

        paths=[]
        distances = []
        #evaporacion de feromonas
        pheromones *= 1 - rho
        #busqueda de un camino para cada hormiga
        for ant in range(ants):

            path,distance = move_ant(positions[ant], graph, pheromones, inv_dist, alpha, beta)
            paths.append(path)
            distances.append(distance)
            if distance <= dist_opt:
                dist_opt = distance
                camino_opt = path
        distpath = [[paths[i],distances[i]] for i in range(len(distances))]
        for k in range(len(paths)):
            for j in range(len(paths[k])-1):
                pheromones[paths[k][j]][paths[k][j+1]] += Q/distances[k]
                pheromones[paths[k][j+1]][paths[k][j]] += Q/distances[k]
            pheromones[paths[k][len(paths[k])-1]][paths[k][0]] += Q/distances[k]
            pheromones[paths[k][0]][paths[k][len(paths[k])-1]] += Q/distances[k]
    return camino_opt,dist_opt

def move_ant(position, graph, pheromones, inv_dist, alpha, beta):

    visited = [position]

    while len(visited) < len(graph):
        
        transiciones = []
        posibles_transiciones = []

        for ciudad in graph:
            if ciudad[0] not in visited:

                transiciones += [pheromones[position][ciudad[0]]**alpha * inv_dist[position][ciudad[0]]**beta]
                posibles_transiciones.append(ciudad[0])

        transiciones = np.array(transiciones)

        next_node = random.choices(posibles_transiciones, weights = transiciones)[0]        
        visited.append(next_node) 
        position = next_node

    distance = sum([dist(graph[visited[i]],graph[visited[i-1]]) for i in range(1,len(visited))])
    distance += dist(graph[len(visited)-1],graph[0])

    return visited,distance            
            
