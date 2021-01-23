import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
             ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
