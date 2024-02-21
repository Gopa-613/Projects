import cv2 #imports the OpenCv library which is used for computer vision task, including image and video processing
face_cap = cv2.CascadeClassifier("C:/Users/SKB PC/AppData/Roaming/Python/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml")
#face_cap captures eyes, nose and features using cascade classifier (haarcascade works on grayscale images)
#xml file contains trained model for face detection (detecting frontal faces here)

video_cap = cv2.VideoCapture(1) # Change the index to 1 for the front camera, captures video
while True: #infinite loop for continuous video processing
    ret, video_data=video_cap.read() #reads single frame from camera
    #video_data stored the actual image data of the frame
    #ret is a boolean variable indicating whether the frame was successfully read
    #multiple assignment function
    col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY) #converts the captured frame to grayscale for detection.(converts color) 
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
        ) #detection parameters
    for(x,y,w,h) in faces: #extracts coordinates and dimensions
        cv2.rectangle(video_data,(x,y),(x+w,y+w),(0,255,0),2) #draws a rectangle around the detected faces
    cv2.imshow("video_live",video_data) #displays the live video in window named "face_detection"

     # Check for user input to end the video
    key = cv2.waitKey(1)

    if key == 27:  # 27 is the ASCII code for the Esc key
        break

    # Check if the window is closed by the user clicking the cross mark
    if cv2.getWindowProperty("video_live", cv2.WND_PROP_VISIBLE) < 1:
        break

    #if cv2.waitKey(10) == ord("a"):
      #  break
video_cap.release()

    