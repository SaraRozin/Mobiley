import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from part3 import SFM
import pickle
matplotlib.use('MacOSX')


class Viewer:

    def __init__(self, pkl_path='../part4/data/pkl_files/dusseldorf_000049.pkl'):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.data = data
        self.focal = data['flx']
        self.pp = data['principle_point']

    def show_distances(self, prev_frame_id, curr_frame_id, prev_container, curr_container):
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, self.focal,
                                                                            self.pp)
        norm_rot_pts = SFM.rotate(norm_prev_pts, R)
        rot_pts = SFM.unnormalize(norm_rot_pts, self.focal, self.pp)
        foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), self.focal, self.pp))
        fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
        prev_sec.set_title('prev(' + str(prev_frame_id) + ')')
        prev_sec.imshow(prev_container.img)
        prev_p = prev_container.traffic_light
        prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')
        curr_sec.set_title('curr(' + str(curr_frame_id) + ')')
        curr_sec.imshow(curr_container.img)
        curr_p = curr_container.traffic_light
        curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')
        for i in range(len(curr_p)):
            curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
            if curr_container.valid[i]:
                curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                              r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
        curr_sec.plot(foe[0], foe[1], 'r+')
        curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        plt.show()
