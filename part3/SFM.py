import math
import numpy as np
from PIL import Image


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = Image.open(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normal = []
    for pt in pts:
        normal.append([(pt[0] - pp[0]) / focal, (pt[1] - pp[1]) / focal])
    return np.array(normal)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    unnormal = []
    for pt in pts:
        unnormal.append((pt[0] * focal + pp[0], pt[1] * focal + pp[1]))
    return np.array(unnormal)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    EM = EM[:-1]
    R = EM[:, :-1]
    tX = EM[0, -1]
    tY = EM[1, -1]
    tZ = EM[2, -1]
    foe = (tX / tZ, tY / tZ)
    return R, foe, tZ


def rot(R, pt):
    return np.dot(R, np.array([pt[0], pt[1], 1]))


def rotate(pts, R):
    # rotate the points - pts using R
    arr_returned = []
    for pt in pts:
        res = rot(R, pt)
        arr_returned.append([res[0] / res[2], res[1] / res[2]])
    return arr_returned


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = ((p[1] * foe[0]) - (foe[1] * p[0])) / (foe[0] - p[0])
    # run over all norm_pts_rot and find the one closest to the epipolar line
    dist = []
    for i, pt in enumerate(norm_pts_rot):
        dist.append((abs((m * pt[0] + n - pt[1]) / (math.sqrt(math.pow(m , 2) + 1))), i))
    ind_min = min(dist)[1]
    # return the closest point and its index
    return ind_min, norm_pts_rot[ind_min]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    Zx = (tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0])
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    Zy = (tZ*(foe[1] - p_rot[1]))/(p_curr[1] - p_rot[1])
    # combine the two estimations and return estimated Z
    x_diff = abs(p_rot[0] - p_curr[0])
    y_diff = abs(p_rot[1] - p_curr[1])
    return (x_diff / (x_diff + y_diff)) * Zx + (y_diff / (x_diff + y_diff)) * Zy
