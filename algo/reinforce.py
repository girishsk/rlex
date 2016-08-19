from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import prettytensor as pt
from space_conversion import SpaceConversionEnv
import tempfile
import sys

class ReinfoceAgent(object):

    def __init__(self, env):
        self.env = env
        self.x = tf.placeholder(
        None
    def act(self, obs, *args):
        None
    def learn(self):
        None