import numpy as np
import torch
import torch.nn as nn
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
BCEcls = nn.BCEWithLogitsLoss()
BCEcls2 = nn.BCELoss()

pred = torch.ones((2,3))
label = torch.ones((2,3))
label[0,0] = 0
pred[0,0] = -20

loss = BCEcls(pred,label)

preds = torch.nn.Sigmoid()(pred)
loss2 = BCEcls2(preds,label)

print(label)
print(preds)

a = torch.log(preds) * label + (1-label) * torch.log(1-preds)

print(a)
print(torch.abs(torch.mean(a)))


print(loss)
print(loss2)

