import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# create aliases for specific modules within Mediapipe
# to simplify their usage later in the code.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# mediapipe use for hand detection,set the mode to static image
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# directory where dataset is stored
DATA_DIR = './data'

data = []
labels = []

# traverse in each sub-dir of ./data
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # iterate each image in directory
        print(img_path)
        print(os.path.join(DATA_DIR, dir_, img_path))
        data_aux = []

        x_ = []
        y_ = []
        # reads image using open cv
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert img to rgb

        results = hands.process(img_rgb) # detect hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x # extracting x,y
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)    # storing x,y coordinates
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
