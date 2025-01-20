import numpy as np
import networkx as nx
import trimesh
import scipy.sparse
import time

# --------------------------------- Utility Functions -------------------------------- #

def create_graph_from_mesh(mesh):
    edge_list = mesh.edges_unique
    vertex_list = mesh.vertices
    graph = nx.Graph()
    for idx, position in enumerate(vertex_list):
        graph.add_node(idx, pos=position)
    for edge in edge_list:
        graph.add_edge(*edge)
    return graph

def extract_boundary(graph, control_points):
    boundary_nodes = []
    for i in range(len(control_points)):
        start = control_points[i]
        end = control_points[(i + 1) % len(control_points)]
        path = nx.shortest_path(graph, start, end)
        boundary_nodes += path[:-1]
    return boundary_nodes

def compute_rw_laplacian(graph):
    lap_matrix = nx.laplacian_matrix(graph)
    adj_matrix = nx.adjacency_matrix(graph)
    inv_deg_matrix = scipy.sparse.diags(1 / adj_matrix.dot(np.ones([adj_matrix.shape[0]])))
    return inv_deg_matrix.dot(lap_matrix)

def find_editable_vertices(graph, boundary, handle_node):
    subgraph = graph.copy()
    subgraph.remove_nodes_from(boundary)
    return list(nx.node_connected_component(subgraph, handle_node))

def visualize_mesh_scene(mesh, boundary_nodes=None, editable_nodes=None):
    scene_objects = [mesh]
    if boundary_nodes:
        scene_objects.append(trimesh.load_path(mesh.vertices[boundary_nodes + [boundary_nodes[0]]]))
    if editable_nodes:
        scene_objects.append(trimesh.points.PointCloud(mesh.vertices[editable_nodes + boundary_nodes]))
    
    return trimesh.Scene(scene_objects)

# -------------------------------- Load Mesh -------------------------------- #
mesh = trimesh.load('./meshes/bunny.ply', process=False)
output_filename = './exports/bunny_%s.ply' % (time.strftime("%Y%m%d%H%M%S", time.localtime()) )

# ---------------------------- Editing Parameters ---------------------------- #
control_handles = {
    24092: [-0.01134, 0.151374, -0.0242688]
}
boundary_control_points = [15617, 24120, 30216, 11236, 6973]

# -------------------------------- Build Graph ------------------------------- #
graph = create_graph_from_mesh(mesh)
boundary_nodes = extract_boundary(graph, boundary_control_points)
editable_vertices = find_editable_vertices(graph, boundary_nodes, list(control_handles.keys())[0])
print(editable_vertices)

# --------------------- Subgraph of the Editable Region --------------------- #
editable_subgraph = graph.subgraph(boundary_nodes + editable_vertices)
node_to_local = {}
for node in editable_subgraph.nodes:
    node_to_local[node] = len(node_to_local)
local_to_node = list(editable_subgraph.nodes)

def get_neighbors(subgraph, node_idx, local_to_node, node_to_local):
    neighbors = []
    for neighbor in subgraph.neighbors(local_to_node[node_idx]):
        neighbors.append(node_to_local[neighbor])
    return neighbors

# -------------------------- Build the Linear System ------------------------- #
laplacian_matrix = compute_rw_laplacian(editable_subgraph).todense()
vertices_matrix = np.matrix([editable_subgraph.nodes[node]['pos'] for node in editable_subgraph.nodes])
delta_matrix = laplacian_matrix.dot(vertices_matrix)
num_nodes = laplacian_matrix.shape[0]

linear_system = np.zeros([3 * num_nodes, 3 * num_nodes])
linear_system[0:num_nodes, 0:num_nodes] = (-1) * laplacian_matrix
linear_system[num_nodes:2*num_nodes, num_nodes:2*num_nodes] = (-1) * laplacian_matrix
linear_system[2*num_nodes:3*num_nodes, 2*num_nodes:3*num_nodes] = (-1) * laplacian_matrix

for i in range(num_nodes):
    neighbor_indices = get_neighbors(editable_subgraph, i, local_to_node, node_to_local)
    neighbor_ring = np.array([i] + neighbor_indices)
    ring_vertices = vertices_matrix[neighbor_ring]
    num_neighbors = ring_vertices.shape[0]
    
    A_matrix = np.zeros([num_neighbors * 3, 7])
    for j in range(num_neighbors):
        A_matrix[j] = [ring_vertices[j, 0], 0, ring_vertices[j, 2], -ring_vertices[j, 1], 1, 0, 0]
        A_matrix[j + num_neighbors] = [ring_vertices[j, 1], -ring_vertices[j, 2], 0, ring_vertices[j, 0], 0, 1, 0]
        A_matrix[j + 2*num_neighbors] = [ring_vertices[j, 2], ring_vertices[j, 1], -ring_vertices[j, 0], 0, 0, 0, 1]
        
    # Moore-Penrose Inversion
    A_pseudoinv = np.linalg.pinv(A_matrix)
    s_vec = A_pseudoinv[0]
    h_vec = A_pseudoinv[1:4]
    t_vec = A_pseudoinv[4:7]

    transformed_delta = np.vstack([
        delta_matrix[i, 0] * s_vec - delta_matrix[i, 1] * h_vec[2] + delta_matrix[i, 2] * h_vec[1],
        delta_matrix[i, 0] * h_vec[2] + delta_matrix[i, 1] * s_vec - delta_matrix[i, 2] * h_vec[0],
        -delta_matrix[i, 0] * h_vec[1] + delta_matrix[i, 1] * h_vec[0] + delta_matrix[i, 2] * s_vec,
    ])
        
    linear_system[i, np.hstack([neighbor_ring, neighbor_ring + num_nodes, neighbor_ring + 2*num_nodes])] += transformed_delta[0]
    linear_system[i + num_nodes, np.hstack([neighbor_ring, neighbor_ring + num_nodes, neighbor_ring + 2*num_nodes])] += transformed_delta[1]
    linear_system[i + 2*num_nodes, np.hstack([neighbor_ring, neighbor_ring + num_nodes, neighbor_ring + 2*num_nodes])] += transformed_delta[2]

# ------------------- Add Constraints to the Linear System ------------------- #
constraint_coefficients = []
constraint_b_values = []

# Boundary constraints
boundary_indices = [node_to_local[i] for i in boundary_control_points]
for idx in boundary_indices:
    constraint_coefficients.append(np.arange(3 * num_nodes) == idx)
    constraint_coefficients.append(np.arange(3 * num_nodes) == idx + num_nodes)
    constraint_coefficients.append(np.arange(3 * num_nodes) == idx + 2 * num_nodes)
    constraint_b_values.append(vertices_matrix[idx, 0])
    constraint_b_values.append(vertices_matrix[idx, 1])
    constraint_b_values.append(vertices_matrix[idx, 2])
    
# Handle constraints
for handle_id, position in control_handles.items():
    idx = node_to_local[handle_id]
    constraint_coefficients.append(np.arange(3 * num_nodes) == idx)
    constraint_coefficients.append(np.arange(3 * num_nodes) == idx + num_nodes)
    constraint_coefficients.append(np.arange(3 * num_nodes) == idx + 2 * num_nodes)
    constraint_b_values.append(position[0])
    constraint_b_values.append(position[1])
    constraint_b_values.append(position[2])
    
constraint_coefficients = np.matrix(constraint_coefficients)
constraint_b_values = np.array(constraint_b_values)

# -------------------------- Solve the Linear System ------------------------- #
A_matrix = np.vstack([linear_system, constraint_coefficients])
b_vector = np.hstack([np.zeros(3 * num_nodes), constraint_b_values])
sparse_A = scipy.sparse.coo_matrix(A_matrix)

solution = scipy.sparse.linalg.lsqr(sparse_A, b_vector)

# -------------------------- Output the Edited Mesh -------------------------- #
edited_vertices = []
for i in range(num_nodes):
    edited_vertices.append([solution[0][i], solution[0][i + num_nodes], solution[0][i + 2 * num_nodes]])
    
edited_mesh = mesh.copy()
for idx, vertex in enumerate(edited_vertices):
    original_idx = local_to_node[idx]
    edited_mesh.vertices[original_idx] = vertex

edited_mesh.export(output_filename)

# --------------------- Overlay Original and Edited Mesh --------------------- #
# Load original mesh and mark as blue
original_bunny = trimesh.load('./meshes/bunny.ply', process=False)
original_bunny.visual.vertex_colors = [0, 0, 255, 100]  # Blue with transparency

# Edited mesh marked as green
edited_mesh.visual.vertex_colors = [0, 255, 0]  # Green with transparency

control_points = original_bunny.vertices[boundary_control_points]
original_handle_point = original_bunny.vertices[list(control_handles.keys())[0]]
current_handle_point = edited_mesh.vertices[list(control_handles.keys())[0]]

control_point_cloud = trimesh.points.PointCloud(control_points, color=np.array([[255, 0, 0]] * len(control_points)))
original_handle_cloud = trimesh.points.PointCloud([original_handle_point], color=np.array([[255, 0, 0]]))  # Original handle point as red
current_handle_cloud = trimesh.points.PointCloud([current_handle_point], color=np.array([[255, 0, 0]]))  # Edited handle point as red

# Create scene with original and edited meshes
scene = trimesh.Scene([original_bunny, edited_mesh, control_point_cloud, original_handle_cloud, current_handle_cloud])

# Visualize overlay
scene.show()
