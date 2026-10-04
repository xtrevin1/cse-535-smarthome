# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import keras
import csv

## import the handfeature extractor class
import handshape_feature_extractor as hfe
import frameextractor as fe
# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

extractor = hfe.HandShapeFeatureExtractor.get_instance()

full_model = extractor.model

inp = keras.Input(shape=(200, 200, 1))
x = inp
for layer in full_model.layers[:-1]:
    x = layer(x)

extractor.model = keras.Model(inputs=inp, outputs=x)


def get_feature_vector(video_path, frames_dir, index):
    fe.frameExtractor(video_path, frames_dir, index)
    frame_path = os.path.join(frames_dir, "%05d.png" % (index + 1))
    image = cv2.imread(frame_path)
    if image is None:
        raise RuntimeError(f"Could not read frame for {video_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return extractor.extract_feature(gray).flatten()

train_dir = "traindata"
train_files = sorted(os.listdir(train_dir))
train_features, train_labels = [], []

for index, file in enumerate(train_files):
    train_features.append(get_feature_vector(os.path.join(train_dir, file), "extracted_training_frames", index))
    train_labels.append(file.split("_")[0]) 


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

test_dir = "test"
test_files = sorted(os.listdir(test_dir))

test_features = [get_feature_vector(os.path.join(test_dir, f), "test_frames", i) for i, f in enumerate(test_files)]


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

predictions = []
for file, vec in zip(test_files, test_features):
    sims = [cosine_similarity(vec, t) for t in train_features]
    best = int(np.argmax(sims))
    predictions.append(train_labels[best])
    print(f"{file} -> {train_labels[best]} ({sims[best]:.4f})")


LABEL_MAP = { "Num0": 0, "Num1": 1, "Num2": 2, "Num3": 3, "Num4": 4, "Num5": 5, "Num6": 6, "Num7": 7, "Num8": 8, "Num9": 9,"FanDown": 10, "FanOff": 11,
    "FanOn": 12,
    "FanUp": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16,
}

unknown = sorted(set(train_labels) - set(LABEL_MAP))
if unknown:
    raise KeyError(f"Add these training label prefixes to LABEL_MAP: {unknown}")

with open("Results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for label in predictions:
        writer.writerow([LABEL_MAP[label]])