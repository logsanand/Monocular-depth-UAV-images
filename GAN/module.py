from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)
