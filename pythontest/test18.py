# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:18:44 2020

@author: admin
"""

# import the necessary packages
import dlib


predictor_path = current_path + "\\model\\shape_predictor_68_face_landmarks.dat"
face_rec_model_path=current_path + "\\model\\face_model.dat'

# 读入模型
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)