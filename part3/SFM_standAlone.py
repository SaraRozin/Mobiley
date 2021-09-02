import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from part3 import SFM



class FrameContainer(object):
    def __init__(self, img_path):
        self.img = Image.open(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []


# read data and run
# curr_frame_id = 25
# prev_frame_id = 24
# pkl_path = '../part4/data/pkl_files/dusseldorf_000049.pkl'
# prev_img_path = '../part4/data/images/dusseldorf_000049_0000' + str(prev_frame_id) + '_leftImg8bit.png'
# curr_img_path = '../part4/data/images/dusseldorf_000049_0000' + str(curr_frame_id) + '_leftImg8bit.png'
# prev_container = FrameContainer(prev_img_path)
# curr_container = FrameContainer(curr_img_path)
# with open(pkl_path, 'rb') as pklfile:
#     data = pickle.load(pklfile, encoding='latin1')
# focal = data['flx']
# pp = data['principle_point']
# prev_container.traffic_light = np.array(data['points_' + str(prev_frame_id)][0])
# curr_container.traffic_light = np.array(data['points_' + str(curr_frame_id)][0])
# EM = np.eye(4)
# for i in range(prev_frame_id, curr_frame_id):
#     EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
# curr_container.EM = EM
# curr_container = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
# visualize(prev_container, curr_container, focal, pp)
