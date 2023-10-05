import pickle
import numpy as np
import cv2
import os
import cvzone

# Open the default camera (usually index 0)
import face_recognition

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
    
# Set the frame dimensions
cap.set(3, 640)
cap.set(4, 480)
imgBackground = cv2.imread('Resources/background.png')

# importing mode imgages
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

print(imgModeList)

# load the encoding file
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds  = encodeListKnownWithIds
print(studentIds)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)


    if not success:
        print("Error: Failed to read frame.")
        break

    # cv2.imshow("Webcam", img)
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print('matches', matches)
        # print('faceDis', faceDis)

        matchIndex = np.argmin(faceDis)
        # print('Match Index', matchIndex)

        if matches[matchIndex]:
            # print('Known face detected')
            # print(studentIds[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)


    cv2.imshow("Face Attendance", imgBackground)
    key = cv2.waitKey(1)  # Wait for a key press for 1 millisecond

    if key == ord('q'):  # Exit the loop if 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
