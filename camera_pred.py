import cv2
import math
import numpy as np
import time
from collections import deque
import digit_recognizer as dr
import LRmodel

cap = cv2.VideoCapture(0)
# collection of points to draw
center_points = deque()

# green colour pointer to be detected
lowergreen = np.array([50, 100, 50])
uppergreen = np.array([90, 255, 255])

# the black board for the models
board = np.zeros((230, 230), dtype='uint8')

while(cap.isOpened()):
    ret, frame = cap.read()
    # flipping the frame
    frame = cv2.flip(frame, 1)
    # applying gaussian blur
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # drawing the rectangle for the board
    cv2.rectangle(frame, (400, 50), (600, 250), (100, 100, 255), 2)
    roi = frame[50:250, 400:600, :]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # detecting colours in the range
    roi_range = cv2.inRange(hsv_roi, lowergreen, uppergreen)
    # applying contours on the detected colours
    image, contours, hierarchy = cv2.findContours(
        roi_range.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # the text to be displayed on the screen
    predict1_text = "Logistic Regression : "
    predict2_text = "CNN Model : "
    # flags to check when drawing started and when stopped
    drawing_started = False
    drawing_stopped = False
    if(len(contours) > 0):
        drawing_started = True
        # getting max contours from the contours
        max_contours = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contours)
        # to avoid divided by zero error
        try:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        except:
            continue
        # center obtained is appended to the deque
        center_points.appendleft(center)
    else:
        drawing_stopped = False
    for i in range(1, len(center_points)):
        if math.sqrt((center_points[i-1][0] - center_points[i][0])**2 +
                     (center_points[i-1][1] - center_points[i][1])**2) < 50:
            cv2.line(roi, center_points[i-1], center_points[i], (200, 200, 200), 5, cv2.LINE_AA)
            cv2.line(board, (center_points[i-1][0]+15, center_points[i-1][1]+15),
                     (center_points[i][0]+15, center_points[i][1]+15), 255, 7, cv2.LINE_AA)
    # the board is resized for the prediction
    input = cv2.resize(board, (28, 28))
    # applying morphological transformation on the drawn digit
    if np.max(board) != 0 and drawing_started == True and drawing_stopped == True:
        kernel = (5, 5)
        input = cv2.morphologyEx(input, cv2.MORPH_OPEN, kernel)
        board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel)
        drawing_started = False
        drawing_stopped = False
    # predicting the digit using LR and CNN
    if np.max(board) != 0:
        LR_input = input.reshape(1, 784)
        test_x = input.reshape((1, 28, 28, 1))
        prediction1 = np.argmax(LRmodel.predict(
            LR_input, dr.LR_params.item().get('weights'), dr.LR_params.item().get('base')))
        prediction2 = np.argmax(dr.model_conv.predict(test_x))
        predict1_text += str(prediction1)
        predict2_text += str(prediction2)
    # displaying the text on the screen
    cv2.putText(frame, predict1_text,
                (5, 420), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, predict2_text,
                (5, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('input', input)
    cv2.imshow('frame', frame)
    cv2.imshow('board', board)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    # clearing the board
    elif k == ord('c'):
        board.fill(0)
        center_points.clear()
cap.release()
cv2.destroyAllWindows()
