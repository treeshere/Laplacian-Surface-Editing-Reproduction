import trimesh
import numpy as np


mesh = trimesh.load('./meshes/bunny.ply')


# marked_vertices = [ 24, 673]


# marked_points = mesh.vertices[marked_vertices]
# marked_points = np.vstack([marked_points, [-0.01134  ,  0.151374 , -0.0242688]])

# point_cloud = trimesh.points.PointCloud(marked_points, colors=[[255, 0, 0]])  # 红色标记
scene = trimesh.Scene([mesh])

scene.show()