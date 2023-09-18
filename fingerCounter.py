import cv2
import time
import os
import HandTrackingModule as htm

wCam,hCam=640,480
cap =cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folder="fingerimg"
myList=os.listdir(folder)
print(myList)
overlayList=[]
for imgpath in myList:
    image=cv2.imread(f'{folder}/{imgpath}')
    overlayList.append(image)
print(len(overlayList))

pTime=0
detector=htm.handDetector(detectionCon=0.75)
while True:


    success, img=cap.read()
    img=detector.findHands(img)
    h,w,c=overlayList[0].shape
    img[0:h,0:w]=overlayList[0]
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS: {int(fps)}',(400,80),cv2.FONT_HERSHEY_PLAIN,
                2,(255,0,5),3)
    cv2.imshow("Image",img)
    key=cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
    




