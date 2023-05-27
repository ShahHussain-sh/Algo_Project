from tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
import sys

from networkx.classes import graph


# input file decider
def i_choice():
    inputFiles = [
    "input10.txt",
    "input20.txt",
    "input30.txt",
    "input40.txt",
    "input50.txt",
    "input60.txt",
    "input70.txt",
    "input80.txt",
    "input90.txt",
    "input100.txt",
]
    print("\n\t\t------ ALGORITHM HUB -------")
    inputchoice=input("\n Press 1 For input10.txt \n Press 2 For input20.txt \n Press 3 For input30.txt \n Press 4 For input40.txt \n Press 5 For input50.txt \n Press 6 For input60.txt \n Press 7 For input70.txt \n Press 8 For input80.txt \n Press 9 For input90.txt \n Press 10 For input100.txt \n\n Press 0 For Exit \n---------\n Enter Your Choice:")
    if inputchoice == '1':
        file_name = inputFiles[0]
        # print(file_name)
        MST_cost=676500000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '2':
        file_name = inputFiles[1]
        # print(file_name)
        MST_cost=555000000
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '3':
        file_name = inputFiles[2]
        # print(file_name)
        MST_cost=2082000000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '4':
        file_name = inputFiles[3]
        # print(file_name)
        MST_cost=3144000000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '5':
        file_name = inputFiles[4]
        # print(file_name)
        MST_cost=3547500000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '6':
        file_name = inputFiles[5]
        # print(file_name)
        MST_cost=4395000000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '7':
        file_name = inputFiles[6]
        # print(file_name)
        MST_cost=4998000000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '8':
        file_name = inputFiles[7]
        # print(file_name)
        MST_cost=6009000000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '9':
        file_name = inputFiles[8]
        # print(file_name)
        MST_cost=6606000000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '10':
        file_name = inputFiles[9]
        # print(file_name)
        MST_cost=7699500000.0
        print(MST_cost)
        return file_name,MST_cost
    elif inputchoice == '0':
        exit()
    

def filing(Input):
    filename = Input
    with open(filename) as f:
        lines = f.readlines()
        lines = (line for line in lines if line)

    count = 0
    list1 = []
    Node = []

    for line in lines:
        count += 1
        if not line.strip():
            continue
        else:
            listli = line.split()
            list1.append(listli)

    v = int(list1[1][0])
    # Adjacenty Matrix
    adjacent = [[0] * v for _ in range(v)]

    # for all nodes 0 to n nodes
    for i in range(0, v):
        ps = (float(list1[2 + i][1]), float(list1[2 + i][2]))
        Node.append(ps)

    # skipping nodes + 2(netsin and num of nodes)
    for i in range(v + 2, len(list1) - 1):
        f = int(list1[i][0])                        #from vertex

        for j in range(1, len(list1[i]), 4):
            t = int(list1[i][j])                    # to vertex
            w = float(list1[i][j + 2])              # weight

            edge = (int(f), int(t), float(w))
            if adjacent[f][t] > w or adjacent[f][t] == 0:
                adjacent[f][t] = w

    source = int(list1[len(list1)-1][0])
    return source,adjacent,v,Node


def returnUndirectedGraph(graph):
    for i in range(0,len(graph)):
        for j in range(0,len(graph[i])):
          if(graph[i][j]!=0 and graph[i][j]!=graph[j][i]):
              min = graph[i][j]
              if(graph[j][i]!=0 and graph[j][i]<min):
                  min = graph[j][i]
              graph[i][j]=min
              graph[j][i]=min
    return graph


def printUnDirectedGraph(adjMat,pos,v):
    g = nx.Graph()

    for i in range(0,v):
        po = (pos[i][0],pos[i][1])
        g.add_node(i,pos=po)

    for i in range(0,v):
        for j in range(0,v):
            if(adjMat[i][j]!=0):
                weight = adjMat[i][j]/10000000
                g.add_edge(i,j,weight=weight)
    weight = nx.get_edge_attributes(g, 'weight')
    pos1 = nx.get_node_attributes(g, 'pos')
    nx.draw_networkx_edge_labels(g, pos1, edge_labels=weight)
    nx.draw(g, pos1, with_labels=1, font_color='black')
    plt.show()


def printDirectedGraph(adjMat,pos):
    g = nx.DiGraph()

    for i in range(0,len(adjMat)):
        po = pos[i]
        g.add_node(i,pos=po)

    for i in range(0,len(adjMat)):
        for j in range(0,len(adjMat[i])):
            if(adjMat[i][j]!=0):
                weight = adjMat[i][j]/10000000
                g.add_edge(i,j,weight = weight)
    weight = nx.get_edge_attributes(g, 'weight')
    pos1 = nx.get_node_attributes(g, 'pos')
    nx.draw_networkx_edge_labels(g, pos1, edge_labels=weight)
    nx.draw(g, pos1, with_labels=1)
    plt.show()

def printInputGraph(i_filename):
    with open(i_filename) as f:
        lines = f.readlines()
        lines = (line for line in lines if line)
    g = nx.DiGraph()
    count = 0
    list = []
    for line in lines:
        count += 1
        if not line.strip():
            continue
        else:
            listli = line.split()
            list.append(listli)
    v = int(list.__getitem__(1).__getitem__(0))
    print(v)
    for i in range(0, v):
        ps = (float(list.__getitem__(2 + i).__getitem__(1)), float(list.__getitem__(2 + i).__getitem__(2)))
        g.add_node(i, pos=ps)
    i = 0
    j = 0
    for i in range(v + 2, len(list) - 1):
        f = int(list.__getitem__(i).__getitem__(0))
        for j in range(1, len(list.__getitem__(i)), 4):
            t = int(list.__getitem__(i).__getitem__(j))
            w = float(list.__getitem__(i).__getitem__(j + 2))
            w=w/10000000
            g.add_edge(f, t, weight=w)
    weight = nx.get_edge_attributes(g, 'weight')
    pos = nx.get_node_attributes(g, 'pos')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=weight)
    nx.draw(g, pos, with_labels=1, font_color='black')
    plt.show()


#   ---------- PRIMS ALGORITHMS

def findMaxVertex(visited, weights,V):
    index = -1
    maxW = -sys.maxsize
    for i in range(V):
        if (visited[i] == False and weights[i] > maxW):
            maxW = weights[i]
            index = i
    return index

def PrimsAlgo(graph,V,S):


    visited = [True] * V
    weights = [0] * V
    parent = [0] * V
    P_mst = 0

    for i in range(V):
        visited[i] = False
        weights[i] = -sys.maxsize

    weights[S] = sys.maxsize
    parent[S] = -1

    for i in range(V - 1):
        maxVertexIndex = findMaxVertex(visited, weights,V)
        visited[maxVertexIndex] = True
        for j in range(V):
            if (graph[j][maxVertexIndex] != 0 and visited[j] == False):
                if (graph[j][maxVertexIndex] > weights[j]):
                    weights[j] = graph[j][maxVertexIndex]
                    parent[j] = maxVertexIndex
    for i in range(V):
            for j in range(V):
                graph[i][j]=0
                if(parent[j]==i):
                    graph[i][j]=weights[j]
                    P_mst += weights[j]
    
    print(graph)
    MST_cost_11=[sum(i) for i in graph]
    # print (su)
    print("MST:",MST_cost_11)
    return graph

#-----------------Kruskal Alorithm

def kunion(i, j,parent):
    a = find(parent,i)
    b = find(parent,j)
    parent[a] = b


def kruskalMST(cost,V):
    G = [[0] * V for _ in range(V)]
    parent=[0]*V

    # Initialize sets of disjoint sets
    for i in range(V):
        parent[i] = i

    # Include minimum weight edges one by one
    edge_count = 0
    K_cost = 0
    while edge_count < V - 1:
        max = -sys.maxsize
        a = -1
        b = -1
        for i in range(V):
            for j in range(V):
                if find(parent,i) != find(parent,j) and cost[i][j] > max and cost[i][j]!=0:
                    max = cost[i][j]
                    a = i
                    b = j
                    K_cost=max
        kunion(a, b,parent)

        print('Edge {}:({}, {}) cost:{}'.format(edge_count, a, b, max))
        
        G[a][b]=max
        G[b][a] = max
        edge_count += 1
        K_cost+=max


    print("MST:",format(K_cost))
    return G

#--------- Bellmen ford

def BellmenFord(graph, V, src):
    dist = [sys.maxsize] * V
    dist[src] = 0
    parent = [-1]*V

    for q in range(V - 1):
        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                x = i
                y = j
                w = graph[i][j]

                if dist[x] + w < dist[y]:
                    dist[y] = dist[x] + w
                    parent[y]=x

    for i in range(V):
        for j in range(V):
            if graph[i][j] == 0:
                continue
            x = i
            y = j
            w = graph[i][j]
            if dist[x] != sys.maxsize and dist[x] + w < dist[y]:
                return None

    for i in range(V):
        for j in range(V):
            graph[i][j]=0

    for i in range(V):
        if(parent[i]!=-1):
            graph[parent[i]][i] = dist[i]
    return graph


# - -------- Dijsktra
def minDistance(V, dist, sptSet):
    min = sys.maxsize
    min_index = -1

    # Search not nearest vertex not in the
    # shortest path tree
    for u in range(V):
        if dist[u] < min and sptSet[u] == False:
            min = dist[u]
            min_index = u

    return min_index


def dijkstra(G, V, src):
    dist = [sys.maxsize] * V
    sptSet = [False] * V
    parent = [-1] * V
    dist[src] = 0
    for cout in range(V):

        x = minDistance(V, dist, sptSet)
        sptSet[x] = True


        for y in range(0, V):

            if (G[x][y] > 0 and sptSet[y] == False) and (dist[y] > dist[x] + G[x][y]):
                dist[y] = dist[x] + G[x][y]
                parent[y] = x
    for i in range(V):
        for j in range(V):
            G[i][j] = 0
    for i in range(V):
        if (parent[i] != -1):
            G[parent[i]][i] = dist[i]
    return G

#------ boruvka

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def boruvka(graph, V):
    bruv_mst = 0
    parent = []
    rank = []
    G  = [[0] * V for _ in range(V)]

    cheapest = []

    numTrees = V
    MSTweight = 0

    for node in range(V):
        parent.append(node)
        rank.append(0)
        cheapest = [-1] * V


    while numTrees > 1:

        for i in range(V):
            for j in range(V):
                if graph[i][j] == 0:
                    continue
                w = graph[i][j]
                set1 = find(parent, i)
                set2 = find(parent, j)

                if set1 != set2:

                    if cheapest[set1] == -1 or cheapest[set1][2] < w:
                        cheapest[set1] = [i, j, w]

                    if cheapest[set2] == -1 or cheapest[set2][2] < w:
                        cheapest[set2] = [i, j, w]
        for node in range(V):


            if cheapest[node] != -1:
                u, v, w = cheapest[node]
                set1 = find(parent, u)
                set2 = find(parent, v)

                if set1 != set2:
                    MSTweight += w
                    union(parent, rank, set1, set2)
                    G[u][v] = w
                    print("Edge %d-%d with weight %d included in MST" % (u, v, w))
                    bruv_mst+=w
                    numTrees = numTrees - 1


        cheapest = [-1] * V
    print(bruv_mst)
    return G


#   floyd warshal

def floydWarshal(graph):
    dist = graph
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(i==j):
                dist[i][j] = 0
            elif(graph[i][j]==0):
                dist[i][j] = sys.maxsize
            else:
                dist[i][j]=graph[i][j]
    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                dist[i][j] = min(dist[i][j],dist[i][k] + dist[k][j])
    for i in range(len(graph)):
        for j in range(len(graph)):
            if(dist[i][j]==sys.maxsize):
                graph[i][j] = 0
            else:
                graph[i][j] = dist[i][j]
    return graph

# Clustering


def Clustering_Coefficient(adjM,p,v):
    g = nx.Graph()

    for i in range(0, v):
        po = (p[i][0], p[i][1])
        g.add_node(i, pos=po)

    for i in range(0, v):
        for j in range(0, v):
            if (adjM[i][j] != 0):
                weight = adjM[i][j]
                g.add_edge(i, j, weight=weight)
    print(nx.average_clustering(g))

def algo_decide(in_filename):
    # s -> source , adjM -> adjency matrix , v -> no_of nodes , p -> position 
    s, adjM, v, p = filing(in_filename)  # reading from file
    
    print("\n\t\t------ ALGORITHM HUB -------")
    algo_choice = input("\n Press 1 --> PRIMS ALGORITHM \n Press 2 --> KRUSKAL ALGORITHM \n Press 3 --> DIJKASTRA ALGORITHM \n Press 4 --> BELLMAN FORD ALGORITHM \n Press 5 --> FLOYD WARSHALL ALGORITHM \n Press 6 --> CLUSTERING COEFFICIENT \n Press 7 --> BORUVKA ALGORITHM \n\n Press 0 --> EXIT \n\n----------------\n ENTER YOUR CHOICE:")
    print("Your Algo choice = ",algo_choice)

    if algo_choice == '1':
        adjM = returnUndirectedGraph(adjM)
        # print("hey")
        # print(adjM)
        adjM = PrimsAlgo(adjM,v,s)
        printUnDirectedGraph(adjM,p,v)
    
    elif algo_choice == '2':
        adjM = returnUndirectedGraph(adjM)
        # print(adjM)
        adjM = kruskalMST(adjM,v)
        printUnDirectedGraph(adjM,p,v)
    
    elif algo_choice == '3':
        adjM = dijkstra(adjM,v,s)
        printDirectedGraph(adjM,p)
    
    elif algo_choice == '4':
        print("s",s)
        print("v",v)
        print("adj",adjM)
        adjM = BellmenFord(adjM, v, s)
        printDirectedGraph(adjM, p)
    
    elif algo_choice == '5':
        adjM = floydWarshal(adjM)
        print(adjM)
        printDirectedGraph(adjM, p)
    
    elif algo_choice == '6':
        adjM = returnUndirectedGraph(adjM)
        Clustering_Coefficient(adjM,p,v)
    
    elif algo_choice == '7':
        adjM = boruvka(adjM,v)
        printDirectedGraph(adjM,p)
    
    elif algo_choice == '0':
        return()

#-----------------------------------------------
# driver

in_filename=i_choice()  # return file name
print("INPUT FILE NAME: ",in_filename)

printInputGraph(in_filename) # ploting input graph 
algo_decide(in_filename)
choice_1=input("Do you want to check more algo`s on same input file \n PRESS 1 --> YES \n PRESS 2 --> NO \n\n ENTER YOUR CHOICE: ")

if choice_1 == '1':
    algo_decide(in_filename)

else:
    choice_11=input("Do you want to check more algo`s on different input file \n PRESS 1 --> YES \n PRESS 2 --> NO \n\n ENTER YOUR CHOICE: ")
    in_filename=i_choice()  # return file name
    print(in_filename)

    printInputGraph(in_filename) # ploting input graph 
    algo_decide(in_filename)
    choice_1=input("Do you want to check more algo`s on same input file \n PRESS 1 --> YES \n PRESS 2 --> NO \n\n ENTER YOUR CHOICE: ")

    if choice_1 == '1':
        algo_decide(in_filename)
    else:
        exit()


    