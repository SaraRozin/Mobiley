from tensorflow.keras.models import load_model

from part1.part1_api import find_tfl_lights
from part3.SFM_standAlone import *
import numpy as np
from PIL import Image
import pickle


class Model:
    def __init__(self, pkl_path='../part4/data/pkl_files/dusseldorf_000049.pkl'):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.data = data
        self.focal = data['flx']
        self.pp = data['principle_point']
        self.loaded_model = load_model("../part2/model2.h5")

    def filter_points(self, image, suspect_x, suspect_y):
        tfl_points = []
        for i in range(len(suspect_x)):
            x, y = suspect_x[i], suspect_y[i]
            l_predictions = self.loaded_model.predict(self.crop_by_x_y(image, x, y).reshape(-1, 81, 81, 3))
            if l_predictions[0][0] > 0.5:
                tfl_points += [(x, y)]
        return tfl_points

    def get_tfl_points(self, image_path, frame_index):
        x_red, y_red, x_green, y_green = find_tfl_lights(np.array(Image.open(image_path)), some_threshold=42)
        suspect_x, suspect_y = x_red + x_green, y_red + y_green
        return self.filter_points(Image.open(image_path), suspect_x, suspect_y)

    def crop_by_x_y(self, image, x, y):
        w, h = image.size[0] - 1, image.size[1] - 1
        left, top, right, bottom = max(0, x - 40), max(y - 40, 0), min(w, x + 41), min(y + 41, h)
        if x - 40 < 0:
            left, right = 0, 81
        if y - 40 < 0:
            top, bottom = 0, 81
        if x + 41 > w:
            left, right = w - 81, w
        if y + 41 > h:
            top, bottom = h - 81, h
        crop_im = image.crop((left, top, right, bottom))
        return np.asarray(crop_im, dtype=np.uint8)

    def get_TFL_distances(self, prev_frame_id, prev_img_path, curr_img_path):
        curr_frame_id = prev_frame_id + 1
        prev_container = FrameContainer(prev_img_path)
        curr_container = FrameContainer(curr_img_path)
        prev_container.traffic_light = np.array(self.get_tfl_points(prev_img_path, prev_frame_id))
        curr_container.traffic_light = np.array(self.get_tfl_points(curr_img_path, curr_frame_id))
        EM = np.eye(4)
        for i in range(prev_frame_id, curr_frame_id):
            EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
        curr_container.EM = EM
        curr_container = SFM.calc_TFL_dist(prev_container, curr_container, self.focal, self.pp)
        return prev_container, curr_container
