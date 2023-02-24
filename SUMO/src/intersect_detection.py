import numpy as np
def intersect_detection(aa, bb):
    """
        Helper function to determine whether there is an intersection between the two polygons described
        by the lists of vertices. Uses the Separating Axis Theorem
        param a an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
        param b an ndarray of connected points [[x_1, y_1], [x_2, y_2],...] that form a closed polygon
        return true if there is any intersection between the 2 polygons, false otherwise
    """
    a = [[aa[0,0], aa[1,0]], [aa[0, 1], aa[1, 1]], [aa[0, 2], aa[1, 2]], [aa[0, 3], aa[1, 3]]]
    b = [[bb[0,0], bb[1,0]], [bb[0, 1], bb[1, 1]], [bb[0, 2], bb[1, 2]], [bb[0, 3], bb[1, 3]]]
    # print(a, b)
    polygons = [a, b]
    minA, maxA, projected, i, i1, j, minB, maxB = None, None, None, None, None, None, None, None

    for i in range(len(polygons)):

        # for each polygon, look at each edge of the polygon, and determine if it separates
        # the two shapes
        polygon = polygons[i]
        for i1 in range(len(polygon)):

            # grab 2 vertices to create an edge
            i2 = (i1 + 1) % len(polygon)
            p1 = polygon[i1]
            p2 = polygon[i2]

            # find the line perpendicular to this edge
            normal = { 'x': p2[1] - p1[1], 'y': p1[0] - p2[0] }

            minA, maxA = None, None
            # for each vertex in the first shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            for j in range(len(a)):
                projected = normal['x'] * a[j][0] + normal['y'] * a[j][1]
                if (minA is None) or (projected < minA): 
                    minA = projected

                if (maxA is None) or (projected > maxA):
                    maxA = projected

            # for each vertex in the second shape, project it onto the line perpendicular to the edge
            # and keep track of the min and max of these values
            minB, maxB = None, None
            for j in range(len(b)): 
                projected = normal['x'] * b[j][0] + normal['y'] * b[j][1]
                if (minB is None) or (projected < minB):
                    minB = projected

                if (maxB is None) or (projected > maxB):
                    maxB = projected

            # if there is no overlap between the projects, the edge we are looking at separates the two
            # polygons, and we know there is no overlap
            if (maxA < minB) or (maxB < minA):
                
                return False
    return True

def rotate_box(angle, center, length, width):
    corners = np.array([[-length/2, -width/2],
                            [length/2, -width/2],
                            [length/2, width/2],
                            [-length/2, width/2]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return np.dot(rot, corners.T)+center[:, None]

def calculate_headings(real_fut_traj_i, cur_i_heading):
    
    diff_x = np.diff(real_fut_traj_i[:, 0])

    diff_y = np.diff(real_fut_traj_i[:, 1])

    headings = np.arctan2(diff_y, diff_x)

    fut_traj_i = np.zeros([len(real_fut_traj_i[:,0]), 3])
    fut_traj_i[:, 0:2] = real_fut_traj_i
    fut_traj_i[0, 2] = cur_i_heading
    fut_traj_i[1:, 2] = headings

    return fut_traj_i