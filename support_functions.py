import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model #used to make our detections
mp_drawing = mp.solutions.drawing_utils # Drawing utilities #to draw the detection

def mediapipe_detection(image,model): #makes the image resolution better
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #color conversion BGR TO RGB
    image.flags.writeable=False                 #Image is no longer writeable
    results=model.process(image)                #to make prediction
    image.flags.writeable=True                  #Image is now writeable
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #color conversion RGB TO BGR
    return image,results

def draw_landmarks(image,results): #helps us show connections
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS) #shows what landmark connects to what landmark
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)



def draw_style_landmarks(image,results): #helps us change the landmarks colors and style them
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,100,10),thickness=1, circle_radius=1), #colors landmark 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) #colors connection
                             ) 
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh])    


def model():
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
    return model()