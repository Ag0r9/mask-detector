from utils import adjust_gamma

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras


def import_face_detection_model():
    return cv2.dnn.readNetFromCaffe('data/face-detector-model/architecture.txt',
                                    'data/face-detector-model/weights.caffemodel')


def import_custom_NN_model():
    return keras.models.load_model('data/Mask_Detection_Model.h5')


def detect(uploaded_file):
    cvNet = import_face_detection_model()
    model = import_custom_NN_model()
    img_size = 124
    gamma = 2.0
    assign = {'0': 'Mask', '1': 'No Mask'}
    image = np.array(uploaded_file)
    image = image[:, :, ::-1].copy()
    image = adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()
    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                im = cv2.resize(frame, (img_size, img_size))
                im = np.array(im) / 255.0
                im = im.reshape(1, 124, 124, 3)
                result = model.predict(im)
                if result > 0.5:
                    label_Y = 1
                else:
                    label_Y = 0
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(image, assign[str(label_Y)], (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (36, 255, 12), 2)

        except:
            pass
    image = image[:, :, ::-1].copy()
    return image