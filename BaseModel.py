#!/usr/bin/env python
# -*- coding: utf-8 -*-
#@Time:  2018/10/10 10:41
#@Author:  jessetjiang
import tensorflow as tf
import random
import numpy as np
import json
import os
import sys
from datetime import date, timedelta
from time import time

class BaseModel(object):
    def __init__(self):
        """
        init the hyperparameters
        """
        pass

    def _init_graph(self):
        '''
        Init graph of tensorflow
        :return:
        '''
        self.graph = tf.Graph()

    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        pass

    def loss(self):
        pass

    def summary(self):
        pass

    def train_one_epoch(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass