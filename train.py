""""Derek Thompson
    University of Washington
    DjTides@cs.washington.edu

    This module is for initiating training for the Diatom Project
"""

import argparse
import sys
import time
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
    Training settings   

    Default values are not tuned do not expect good results from default values
"""



