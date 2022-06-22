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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

### Gather configuration parameters
def gather_arg():

    conf_par = configparser.ConfigParser()
    try:
        conf_par.read('credentials.ini')
        host= conf_par.get('camera', 'host')
        broker = conf_par.get('mqtt', 'broker')
        port = conf_par.getint('mqtt', 'port')
        prototxt = conf_par.get('ssd', 'prototxt')
        model = conf_par.get('ssd', 'model')
        conf = conf_par.getfloat('ssd', 'conf')
    except:
        print('Missing credentials or input file!')
        sys.exit(2)
    return host, broker, port, prototxt, model, conf

## connect to MQTT Broker ###
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        global Connected                #Use global variable
        Connected = True                #Signal connection
    else:
        print("Connection failed")

(host, broker, port, prototxt, model, conf) = gather_arg()
print(host)
Connected = False   #global variable for the state of the connection
client = mqttClient.Client("Python")               #create new instance
client.on_connect= on_connect                      #attach function to callback
client.connect(broker, port=port)          #connect to broker
client.loop_start()        #start the loop
while Connected != True:    #Wait for connection
    time.sleep(1.0)

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
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, result):
    # Draw face connections
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 

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
    face = tf.constant([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]) \
        if result.face_landmarks else np.zeros(468*3)
    face = tf.cast(tf.reshape(face, [-1]), tf.float32)
    lh = tf.constant([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]) \
        if result.left_hand_landmarks else np.zeros(21*3)
    lh = tf.cast(tf.reshape(lh, [-1]), tf.float32) 
    rh = tf.constant([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]) \
        if result.right_hand_landmarks else np.zeros(21*3)
    rh = tf.cast(tf.reshape(rh, [-1]), tf.float32)
    pose = tf.cast(pose, tf.float32)
    face = tf.cast(face, tf.float32)
    
    return tf.concat([pose, face, lh, rh], axis=0)


# Forlders for data collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('data-collection/MP_Data')

# Actions that we try to detect
actions = np.array(['Falling', 'Not falling'])

# Thirty videos worth of data
no_sequences = 5

# Videos are going to be 30 frames in length
sequence_length = 24


for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


cap = cv2.VideoCapture(host)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Loop through actions
    for action in actions:
        #Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length
            for frame_num in range(sequence_length):


                # Read the feed
                ret, frame = cap.read()

                # Make detections
                image, result = mediapipe_detection(frame, holistic)
                # print(result)

                # Draw landmarks
                # draw_styled_landmarks(image, result)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video number {}'.format(action, sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                # New exports
                # keypoints = extract_keypoints(result)
                
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                # np.save(npy_path, keypoints)

                # Show to screen
                cv2.imshow('Fall Detection', image)
        

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            time.sleep(5)
    cap.release()
    cv2.destroyAllWindows()