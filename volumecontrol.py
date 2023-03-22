import mediapipe as mp
import cv2
from math import hypot
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
mphands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
Hands=mphands.Hands(min_detection_confidence=0.5,max_num_hands=1)
cap=cv2.VideoCapture(0)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol=volume.GetVolumeRange()
# print(vol)
minv=vol[0]
maxv=vol[1]
# vol1=0
# vol2=400
# volper=0
while True:
    data,image=cap.read()
    image=cv2.flip(image,1)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=Hands.process(image)
    #print(results)
    lmlist=[]
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id,lm in enumerate(hand_landmarks.landmark):
                height,width,c=image.shape
                cx=int(lm.x*width)
                cy=int(lm.y*height)
                lmlist.append([id,cx,cy])
            mp_drawing.draw_landmarks(image,hand_landmarks,mphands.HAND_CONNECTIONS)
            # print(lmlist)
    if len(lmlist)!=0 and len(lmlist)==21:
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2,y2=lmlist[8][1],lmlist[8][2]
        # print(x1,y1)
        cv2.circle(image,(x1,y1),6,(255,0,0),cv2.FILLED)
        cv2.circle(image,(x2,y2),6,(255,0,0),cv2.FILLED)
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
        l=hypot(x2-x1,y2-y1)
        vol1=np.interp(l,[10,200],[minv,maxv])
        # vol2=np.interp(l,[10,250],[400,100])
        # volper=np.interp(l,[10,250],[0,100])
        
        # print(vol)
        if lmlist[20][2]>lmlist[18][2]: 
            volume.SetMasterVolumeLevel(vol1, None)
    # cv2.rectangle(image,(50,100),(85,400),(90,255,0),3)
    # cv2.rectangle(image,(50,int(vol2)),(85,400),(90,255,0),cv2.FILLED)
    # cv2.putText(image,f'{int(volper)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        # # # # # blevel=np.interp(l,[])
        # print(l)
    
        
    

            



    cv2.imshow('hand tracking',image)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()