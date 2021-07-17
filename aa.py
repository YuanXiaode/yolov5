import numpy as np
import torch
import torchvision
import multiprocessing as mp
from pprint import pprint
from PIL import Image
from pathlib import Path
import re
import platform
import os
from itertools import repeat
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import glob
import random
import math
import pandas as pd
# import seaborn as sns
import argparse
import datetime
# import albumentations as A
import cv2 as cv

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

shape1 = []
shape1.append((416,288))
shape1.append((416,250))
shape1.append((416,249))
shape1.append((410,299))

print(np.stack(shape1, 0).max(0))

shape1 = [make_divisible(x, 32) for x in np.stack(shape1, 0).max(0)]
