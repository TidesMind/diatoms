""" Derek Thompson
    University of Washington
    Diatom project
    This dataset module is a test set for setting up project and test project specific methods
"""

from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

"""
   This class is a torch dataset for testing for the diatom project  
"""


class TestSet(Dataset):

    def __init__(self, train=True, transform=None):
        self.transforms = transform
        if train:
            self.data_info = pd.read_csv('/media/tides/SSD1/diatom_proj/diatoms/data/csv_files/t_train.csv', header=None)
        else:
            self.data_info = pd.read_csv('/media/tides/SSD1/diatom_proj/diatoms/data/csv_files/t_test.csv', header=None)

        self.img_arr = np.asarray(self.data_info.iloc[:, 0])
