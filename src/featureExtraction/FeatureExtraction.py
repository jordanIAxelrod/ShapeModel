import numpy as np
import open3d


# Create a bounding box of each image.
# Sort the lengths of the sides.
# Find mean and variance of the different sides
def find_bounding_box(shapes):
    bounding_boxes = []
    point_clouds = []
    for k in range(shapes.shape[0]):
        point_cloud = open3d.geometry.PointCloud()
        points = open3d.utility.Vector3dVector(shapes[k])
        point_cloud.points = points
        point_clouds.append(point_cloud)
        obb = point_cloud.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        bounding_boxes.append(obb)
    return bounding_boxes, point_clouds

def get_bounding_box_stats(bounding_boxes):
    sizes = []
    stats = {}
    letters = ['L', 'M', 'S']
    for size in range(3):
        sizes.append(np.array([box.extent[size] for box in bounding_boxes]))
        stats[letters[size]] = {}
        stats[letters[size]]['Mean'] = np.mean(sizes[size])
        stats[letters[size]]['Variance'] = np.var(sizes[size])
    return stats

# Find principal moment of inertia
# Find perpendicular plane that has half of the points on each side
# Find two median points in the second principal moment
# find the direction between them.
# project the points onto this direction and find the
# max and min. This is the height.
# This seems less stable for other directions


if __name__ == "__main__":
    shapes = Results.read_data(r"C:\Users\jda_s\Box\bone_project\heart_dataset\masks")
    print(1)
    bounding_boxes, point_clouds = find_bounding_box(shapes)
    print(2)
    stats = get_bounding_box_stats(bounding_boxes)
    print(3)
    print(stats)
    open3d.visualization.draw_geometries([point_clouds[0], bounding_boxes[0]])
