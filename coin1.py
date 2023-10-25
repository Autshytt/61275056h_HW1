import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
kernel = np.ones ((3,3),np.uint8)
kernel_b = (7,7)


def gaussian_weighted_rgb(image, sigma):
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 创建一个高斯核
    kernel = cv2.getGaussianKernel(height, sigma)
    gaussian_kernel = kernel * kernel.T

    # 扩展高斯核以匹配图像尺寸
    gaussian_kernel = cv2.resize(gaussian_kernel, (width, height))

    # 使用高斯核对每个通道进行加权平均
    weighted_r = np.sum(image[:, :, 0] * gaussian_kernel) / np.sum(gaussian_kernel)
    weighted_g = np.sum(image[:, :, 1] * gaussian_kernel) / np.sum(gaussian_kernel)
    weighted_b = np.sum(image[:, :, 2] * gaussian_kernel) / np.sum(gaussian_kernel)

    return weighted_r, weighted_g, weighted_b

def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度

    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮

    if f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return '0'
    else:
        return ''

cap = cv2.VideoCapture(0)            # 讀取攝影機
fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv2.LINE_AA               # 印出文字的邊框
rew,reh = 900,600
sigma = 0
# mediapipe 啟用偵測手掌
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
                             
    while True:
        ret, img = cap.read()
        frame = img.copy()
        img_contour=  frame.copy()
        img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img_gray,55,250)
        #new_canny = cv2.morphologyEx(canny,cv2.MORPH_OPEN,kernel)
        #new_canny = cv2.erode(canny,kernel,iterations =2) 
        #new_canny = cv2.dilate(canny,kernel,iterations= 3)
        #new_canny = cv2.erode(new_canny,kernel,iterations =2) 
        #new_canny = cv2.dilate(new_canny,kernel,iterations= 2)
        
        contours, hierarchy =  cv2.findContours(canny, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
        #contours, hierarchy =  cv2.findContours(canny, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
        img_re= cv2.resize(img, (rew,reh))                 # 縮小尺寸，加快處理效率
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img_re, cv2.COLOR_BGR2RGB)  # 轉換成 RGB 色彩
        results = hands.process(img2)                # 偵測手勢
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = []                   # 記錄手指節點座標的串列
                for i in hand_landmarks.landmark:
                    # 將 21 個節點換算成座標，記錄到 finger_points
                    x = i.x*rew
                    y = i.y*reh
                    finger_points.append((x,y))
                if finger_points:
                    finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                    #print(finger_angle)                     # 印出角度 ( 有需要就開啟註解 )
                    text = hand_pos(finger_angle)            # 取得手勢所回傳的內容
                    cv2.putText(img_re, text, (30,120), fontFace, 5, (255,255,255), 10, lineType) # 印出文字
                    if text == '0':
                      #while text =='1' or text =='
                        for i in contours:
                            weighted_rgb = gaussian_weighted_rgb(img, sigma)
                            #print(i)
                            area=cv2.contourArea(i)
                            print(area)
                            if  area > 600  and  area < 6000:
                                peri =cv2.arcLength(i,True)
                                ver=cv2.approxPolyDP(i,peri*0.02,True)
                                cv2.drawContours(img_contour, i , -1 ,(0,0,0),4 )
                                #print(len(ver))
                                x,y,w,h = cv2.boundingRect(ver)
                                #print (w,h)
                                cv2.rectangle(img_contour,(x-10,y-10),(x+w+10,y+h+10), weighted_rgb,cv2.FILLED)
                                corners=len(ver)
                                #if corners >= 7 and corners<=13:
                                        #cv2.putText(img_contour,'coin',(x,y-5),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,0),2)
                    
                     
        #cv2.imshow('reality',frame)    
        #cv2.imshow('gray',img_gray)
        cv2.imshow('canny',canny)
        #cv2.imshow('new_',new_canny)
        cv2.imshow('contour', img_contour)
        cv2.imshow('hand', img_re)
        #cv2.imshow('b',background)
        
        if cv2.waitKey(2)  == 27 :
            break
cap.release()
cv2.destroyAllWindows()