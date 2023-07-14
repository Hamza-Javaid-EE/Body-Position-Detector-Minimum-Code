import cv2  #Image processing
import mediapipe as mp  #Framework to get pose estimation
import time

cap = cv2.VideoCapture(0)
pTime = 0


#Drawing Points
mpDraw = mp.solutions.drawing_utils
#Pose Detection
mpPose = mp.solutions.pose
pose= mpPose.Pose()
while True:
    success, img = cap.read()
    #Conversion because mediapipe works on RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS,landmark_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
                              )
        #Each id/point
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx, cy = int(lm.x *w), int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)


    # Resize the frame to the desired width and height
    #img = cv2.resize(img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)  # 1 millisecond delay