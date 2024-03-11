import numpy as np
import copy
import torch

class BoxCreator(object):
    def __init__(self):
        self.box_list = []

    def reset(self):
        self.box_list.clear()

    def generate_box_size(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.box_list) < length:
            self.generate_box_size()
        return copy.deepcopy(self.box_list[:length])

    def drop_box(self):
        assert len(self.box_list) >= 0
        self.box_list.pop(0)

class RandomBoxCreator(BoxCreator):
    default_box_set = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                default_box_set.append((2+i, 2+j, 2+k))

    def __init__(self, box_size_set=None):
        super().__init__()
        # self.box_set = box_size_set
        # if self.box_set is None:
        #     self.box_set = RandomBoxCreator.default_box_set
        
        # TODO: this is a hacking code to use an existing series of box sizes for random creator
        self.box_set = [[8, 10, 14], [8, 14, 10], [10, 8, 14], [10, 14, 8], [14, 8, 10], [14, 10, 8], [6, 8, 14], [6, 14, 8], [8, 6, 14], [8, 14, 6], [14, 6, 8], [14, 8, 6], [6, 7, 11], [6, 11, 7], [7, 6, 11], [7, 11, 6], [11, 6, 7], [11, 7, 6], [5, 6, 9], [5, 9, 6], [6, 5, 9], [6, 9, 5], [9, 5, 6], [9, 6, 5], [5, 5, 8], [5, 8, 5], [8, 5, 5], [4, 5, 7], [4, 7, 5], [5, 4, 7], [5, 7, 4], [7, 4, 5], [7, 5, 4]]

        print(self.box_set)

    def generate_box_size(self, **kwargs):
        idx = np.random.randint(0, len(self.box_set))
        self.box_list.append(self.box_set[idx])

class LoadBoxCreator(BoxCreator):
    def __init__(self, data_name=None):
        super().__init__()
        self.data_name = data_name
        print("load data set successfully!")
        self.index = 0
        self.box_index = 0
        self.traj_nums = len(torch.load(self.data_name))

    def reset(self, index=None):
        self.box_list.clear()
        box_trajs = torch.load(self.data_name)
        self.recorder = []
        if index is None:
            self.index += 1
        else:
            self.index = index
        self.boxes = box_trajs[self.index]
        self.box_index = 0
        self.box_set = self.boxes
        self.box_set.append([10, 10, 10])

    def generate_box_size(self, **kwargs):
        if self.box_index < len(self.box_set):
            self.box_list.append(self.box_set[self.box_index])
            self.recorder.append(self.box_set[self.box_index])
            self.box_index += 1
        else:
            self.box_list.append((10, 10, 10))
            self.recorder.append((10, 10, 10))
            self.box_index += 1

class RealBoxCreator(BoxCreator):
    def __init__(self):
        super().__init__()
        print("[REAL] start to load real box data ...")

    def generate_box_size(self, **kwargs):
        box_size = self.rp_get_box()
        self.box_list.append(box_size)
        
    def rp_get_box(self):
        """
            Subscribe the box l, w, h from camera
            Currently using a fake one
        """
        # TODO: use real data from camera instead of fake data
        box_size = np.random.randint(80, 230, 3)
        box_grid_size = self.boxsize_to_grid(box_size)
        return box_grid_size
    
    def boxsize_to_grid(self, box_size, pallet_size=1000, pallet_grid_num=25):
        """
            Convert a box's actual size to grid numbers
        """
        grid_size = pallet_size / pallet_grid_num
        box_grid_num = np.ceil((np.array(box_size)/grid_size)).astype(int).tolist()
        return box_grid_num