#!/usr/bin/python
# -*- coding: utf-8 -*-

def _init():
    global num_layer_list
    global currentdepth
    global updownload
    global priorStageFactor
    global whenEspcn
    num_layer_list = []
    currentdepth = 0
    updownload = {}
    priorStageFactor = 0