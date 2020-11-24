# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 23:14:48 2020

@author: admin
"""

import importlib

params = importlib.import_module('b.c.c') #绝对导入
params_ = importlib.import_module('.c.c',package='b') #相对导入