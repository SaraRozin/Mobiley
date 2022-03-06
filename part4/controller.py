from model import Model
from view import Viewer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Controller:

    def __init__(self, pls_path="play_list.pls"):
        with open(pls_path) as pls_file:
            self.data = pls_file.readlines()

    def controller(self):
        frames = self.data
        index_frame, distances = int(frames[1]), []
        for i in range(len(frames))[2::2]:
            tfl_man = Model()
            v = Viewer()
            prev_image, cur_image = frames[i].split('\n')[0], frames[i + 1].split('\n')[0]
            containers = tfl_man.get_TFL_distances(index_frame, prev_image, cur_image)
            distances += [containers[1].traffic_lights_3d_location[:, 2]]
            v.show_distances(index_frame, index_frame + 1, containers[0], containers[1])
            index_frame += 2
        return distances


if __name__ == '__main__':
    c = Controller()
    c.controller()
