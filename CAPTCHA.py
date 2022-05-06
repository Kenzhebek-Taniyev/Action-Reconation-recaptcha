from extract import mediapipe_landmarks
import mediapipe as mp
import cv2
import numpy as np
import os
import tensorflow as tf
import random
from collections import Counter

model = tf.keras.models.load_model(os.path.join('models', 'action_16_classes_3.h5'))
classes = np.array([clas[:-4] for clas in os.listdir('types of actions')]) 
holistic = mp.solutions.holistic
cut = [np.zeros((1662, )) for _ in range(30)]                            # Remember last 30 frames to predict

# Creating test case
case = random.randint(0, len(classes)-1)
test_case = [case]
for _ in range(4):
    while test_case[-1] == case:
        case = random.randint(0, len(classes)-1)
    test_case.append(case)
print(test_case)
test_pass = True

# Starts reading your face
cap = cv2.VideoCapture(0)
with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as mp:
    while cap.isOpened():
        if not test_case:                                                # Pops the test case if is done, and breaks at the end (also ending the program)
            break
        if test_pass:
            case = test_case.pop()
            test_pass = False
        
        _, frame = cap.read()
        image, landmarks = mediapipe_landmarks(frame, mp)                # again, get keypoints
        
        cut.append(landmarks)
        cut = cut[-30:]                                                  # Keep only last 30 frames

        res = model.predict(np.expand_dims(cut, axis=0))[0]
        
        if res[np.argmax(res)] > 0.2 and np.argmax(res) == case:         # Check if satisfies the test case
            test_pass = True

            
#         put probability table        
        for num, prob in enumerate(res):
            cv2.rectangle(image, (0, 60+num*25), (int(prob*100), 90+num*25), (24, 17, 166), -1)
            cv2.putText(image, classes[num], (0, 85+num*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

#         How many of tests are passed
        cv2.putText(image, classes[case], (540, 200),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'{4-len(test_case)} of 4 is passed', (350, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
#         Putting tutorial image to right bottom
        tutorial_image = os.path.join('types of actions', classes[case]+'.png')
        tutorial_image = cv2.imread(tutorial_image, cv2.IMREAD_COLOR)
        tutorial_image = cv2.resize(tutorial_image, (150, 120))
        h, w, _ = tutorial_image.shape
        image[360:h+360, 490:w+490] = tutorial_image
        
        cv2.imshow('OpenCV Feed', image)
               
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()                # Illuminates are watching
    cv2.destroyAllWindows()
    
print('All test cases successfully passed.')