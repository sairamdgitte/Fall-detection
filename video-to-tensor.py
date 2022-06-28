from re import sub
import time, os

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

import sys
import configparser
import time
import numpy as np
import imutils
import cv2
import paho.mqtt.client as mqttClient
from matplotlib import pyplot as plt
import time
import mediapipe as mp


# Keyponits
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion BGR to RGB
    image.flags.writeable = False                   # image is no longer writeable
    result = model.process(image)                   # Make prediction
    image.flags.writeable = True                    # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion RGB to BGR

    return image, result


def draw_landmarks(image, result):
    # mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, result):
    # # Draw face connections
    # mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #                         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                         mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 

    # Draw pose connections
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 


# Keypoints extract values
def extract_keypoints(result):
    pose = tf.constant([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]) \
    if result.pose_landmarks else np.zeros(33*4)
    pose = tf.cast(tf.reshape(pose, [-1]), tf.float32)
    # face = tf.constant([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]) \
    #     if result.face_landmarks else np.zeros(468*3)
    # face = tf.cast(tf.reshape(face, [-1]), tf.float32)
    lh = tf.constant([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]) \
        if result.left_hand_landmarks else np.zeros(21*3)
    lh = tf.cast(tf.reshape(lh, [-1]), tf.float32) 
    rh = tf.constant([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]) \
        if result.right_hand_landmarks else np.zeros(21*3)
    rh = tf.cast(tf.reshape(rh, [-1]), tf.float32)
    # pose = tf.cast(pose, tf.float32)
    # face = tf.cast(face, tf.float32)
    
    return tf.concat([pose, lh, rh], axis=0)

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['Forward', 'Stop', 'Rotate'])

# Thirty videos worth of data
no_sequences = 120

# Videos are going to be 30 frames in length
sequence_length = 30

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions that we try to detect
actions = np.array(['Falling', 'Not falling'])

# Thirty videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
sequence_length = 30




# Get number of frames in each video
# falling_frames = int(falling.get(cv2.CAP_PROP_FPS))
# print(cv2.__version__)
# print(falling_frames)

for action in actions:
    for sequence in range(30):
        try:
            os.makedirs(os.path.join('Float', action, str(sequence)))
        except:
            pass


# Create VideoCapture objects
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

colors = [(245,117,16), (117,245,16), (16,117,245)]


# Create function to read all frames of a video in an array
def read_frames(folder, action):
    """
    INPUTS:
    video_capture: an OpenCV VideoCapture object whose frames we   want to read
    max_frames: the maximum number of frames we want to read
    
    OUTPUT:
    array of all the frames until max_frames
    """
    # Initialize empty array
    frames_array = []
    
    # Keep track of the frame number
    # frame_nb = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # iterate through the frames and append them to the array
        
        # for sequence in range(no_sequences):
        #     for frame_num in range(sequence_length):
    # while video_capture.isOpened() and frame_nb < max_frames:
        # frame = video_capture.read()


        for i in range(len(os.listdir(folder))):
            print(os.listdir(folder)[i])
            subfolder = os.listdir(folder)[i]
            for j in range(len(os.listdir(folder+'/{}'.format(subfolder)))):
                # print(j)
                frame = cv2.imread(folder+'/'+subfolder+'/'+os.listdir(folder+'/{}'.format(subfolder))[j])
                
                # Make detections
                image, result = mediapipe_detection(frame, holistic)
                # print(result)

                # Draw landmarks
                draw_styled_landmarks(image, result)
                # New exports
                keypoints = extract_keypoints(result)
                # print(keypoints.shape)
                npy_path = os.path.join('Float', action, str(i), str(j))
                np.save(npy_path, keypoints)
                # Show to screen
                cv2.imshow('Action Detection', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass
        # frame_nb += 1

    # release the video capture
    # video_capture.release()
    cv2.destroyAllWindows()
    
    # return the array
    # return (res, image)


for i in os.listdir('Data'):
    # print('MP_Data'+ '/'+ i)
    print(i)
    # for j in os.listdir('Data'+'/'+i+'/'+i):
        # print(j)
    # falling = cv2.imread('Data'+'/'+i+'/'+i+'/')
    
    falling_array = read_frames(folder='Data'+'/'+i, action=i)