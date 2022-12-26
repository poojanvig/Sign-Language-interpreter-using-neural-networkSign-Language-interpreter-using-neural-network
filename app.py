from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import time
from imutils.video import VideoStream
import imutils
import mediapipe as mp
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential #to build sequential nueral network
from tensorflow.keras.layers import LSTM, Dense #action detection
from tensorflow.keras.callbacks import TensorBoard #to trace and monitor our model as it is being trained
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

actions=np.array(['hello','thanks','morning','how are you','fine'])
model=Sequential()
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662)))
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('action.h5')
# custom imports
from support_functions import (mediapipe_detection,
draw_style_landmarks,
extract_keypoints)




app = Flask(__name__)

DATA_PATH=os.path.join('MP_Data')
mp_holistic = mp.solutions.holistic # Holistic model #used to make our detections
mp_drawing = mp.solutions.drawing_utils # Drawing utilities #to draw the detection
#Actions we try to detect
actions=np.array(['hello','thanks','morning','how are you','fine'])
#Forty Videos worth data
no_sequences=40
#videos are going to be 30 frames in length
sequence_length=30
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
        except:
            pass

label_map={label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    

    cap = cv2.VideoCapture(0)
    #test in real time
    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    while True:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                draw_style_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    print(res)
                    print(res[np.argmax(res)], threshold)

                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                        print(f"sentence: {sentence}")
                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                    
                    cv2.rectangle(image, (0,0), (640, 40), (0,0,0), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                # cv2.imshow('OpenCV Feed', image)
                
                # Break gracefully
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                    # break
                
                    ret, buffer = cv2.imencode('.jpg', image)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                     

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)