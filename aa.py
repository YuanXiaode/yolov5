import numpy as np
import torch
import torchvision
import multiprocessing as mp
from pprint import pprint
from PIL import Image
from pathlib import Path
import re
import platform
import pkg_resources as pkg
import os
from itertools import repeat
import time
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import glob
import random
import math
import pandas as pd
import seaborn as sns
import argparse
import datetime

a = np.array([1,2,3])
print(a)
print(a[:])
print(a[[1,1,2]])