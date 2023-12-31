import random
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt

import time
import statistics

def generate_random_graph(num_nodes, max_edges_per_node):
  graph = {}
  nodes = [chr(ord('A') + i) for i in range(num_nodes)]

  for node in nodes:
    num_edges = random.randint(1, max_edges_per_node)

    random_edges = random.sample(nodes, num_edges)
    random_edges = [neighbor for neighbor in random_edges if neighbor != node]

    graph.setdefault(node, [])
    graph[node] += [
        element for element in random_edges if element not in graph[node]
    ]

    for neighbor in random_edges:
      if node not in graph.get(neighbor, []):
        graph.setdefault(neighbor, []).append(node)
  return graph

def parse_graph(graph_dict):
  edges = set()

  for node, neighbors in graph_dict.items():
    for neighbor in neighbors:
      edges.add((node, neighbor))

  return edges

def wavefront_algorithm(graph, start, end):
  wavefront = {vertex: None for vertex in graph}
  wavefront[start] = 0

  queue = deque([start])

  while queue:
    current_vertex = queue.popleft()
    if (current_vertex not in graph):
      continue
    for neighbor in graph[current_vertex]:
      if wavefront[neighbor] is None:
        wavefront[neighbor] = wavefront[current_vertex] + 1
        queue.append(neighbor)
  path = []
  current_vertex = end
  while current_vertex is not None:
    path.insert(0, current_vertex)
    current_vertex = None if wavefront.get(current_vertex) is None else next(
        (v for v in graph.get(current_vertex, [])
         if wavefront.get(v) == wavefront.get(current_vertex, 0) - 1), None)

  return path

num_vertices = 8
max_edges_per_node = 3

start_vertex = chr(ord('A') + random.randint(0, num_vertices))
end_vertex = chr(ord('A') + random.randint(0, num_vertices))

random_graph = generate_random_graph(num_vertices, max_edges_per_node)



shortest_path = wavefront_algorithm(random_graph, start_vertex, end_vertex)
print(f"Shortest path from {start_vertex} to {end_vertex}: {shortest_path}")







# Визуализация графа
def visualize_graph(edges_set):
  G = nx.Graph()
  G.add_edges_from(edges_set)

  pos = nx.spring_layout(G)  
  nx.draw(G,
          pos,
          with_labels=True,
          font_weight='bold',
          node_size=700,
          node_color='skyblue',
          font_color='black',
          font_size=8)

  nx.draw_networkx_edges(G,
                         pos,
                         edgelist=edges_set,
                         width=2,
                         edge_color='gray')

  plt.show()

edges_set = parse_graph(random_graph)
visualize_graph(edges_set)




def measure_execution_time(graph_generator, num_nodes, max_edges_per_node, num_trials=10):
  execution_times = []

  for _ in range(num_trials):
      graph = graph_generator(num_nodes, max_edges_per_node)
      start_time = time.time_ns()
      # Здесь вызывается ваш алгоритм с графом graph
      shortest_path = wavefront_algorithm(graph, start_vertex, end_vertex)
      end_time = time.time_ns()
      execution_times.append(end_time - start_time)

  # Выборка значений между квантилем порядка 0.20 и квантилем порядка 0.80
  lower_quantile = statistics.quantiles(execution_times, n=5)[1]
  upper_quantile = statistics.quantiles(execution_times, n=5)[3]
  filtered_times = [t for t in execution_times if lower_quantile <= t <= upper_quantile]

  # Вычисление среднего времени выполнения
  average_time = sum(filtered_times) / len(filtered_times)

  return average_time

# Пример использования
num_nodes = 10
max_edges_per_node = 4

average_time = measure_execution_time(generate_random_graph, num_nodes, max_edges_per_node)
print(f"Average execution time for {num_nodes} nodes: {average_time} ns")