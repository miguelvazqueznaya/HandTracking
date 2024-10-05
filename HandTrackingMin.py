import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #chooses which webcam to capture.

mpHands = mp.solutions.hands
hands = mpHands.Hands()  #creates an object hands from mediapipe hands using the default parameters
mpDraw = mp.solutions.drawing_utils #uses the drawing method to draw the points on a hand that are tracked

pTime = 0
cTime = 0

while True:         #opens webcam and runs it
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converts image to RGB because hands object only uses RGB
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape #Retrieves image dimensions before loop
            for id, lm in enumerate(handLms.landmark): #loops through detected hand landmarks
                #print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y *h) #converts coordinates to pixel values
                print(id, cx, cy)
                if id==0: #talking about first landmark
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)   #doesn't use imgRGB because we are displaying original image.
                                                                            #handLms is a single hand
                                                                            #mpHands.HAND_CONN draws the connections
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255,), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)