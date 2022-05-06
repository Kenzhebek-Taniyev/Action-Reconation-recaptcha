def mediapipe_landmarks(image, model):
    """
    Detects mediapipe landmarks and returns.
    
    Arguments:
        image: numpy.ndarray     - frame
        model: model             - for processing image
    
    Returns:
        image: numpy.ndarray     - frame with landmarks drawn
        landmarks: numpy.ndarray - dataset of landmarks 

    """
    import mediapipe as mp
    import cv2
    import numpy as np
    import os
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    holistic = mp.solutions.holistic
    drawing_utils = mp.solutions.drawing_utils
    drawing_utils.draw_landmarks(image, results.left_hand_landmarks, holistic.HAND_CONNECTIONS,
                              drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
                              drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1))
    drawing_utils.draw_landmarks(image, results.right_hand_landmarks, holistic.HAND_CONNECTIONS,
                              drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                              drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

    right_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    left_hand_landmarks = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    pose_landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face_landmarks = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    landmarks = np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks])
    
    
    return image, landmarks