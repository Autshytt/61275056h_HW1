import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    # 将图像转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 100, 250)

    # 查找物体轮廓
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # 近似轮廓
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 绘制近似轮廓
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        
        # 获取近似轮廓的顶点坐标
        for point in approx:
            x, y = point[0]
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            print (x,y)
            x,y,w,h = cv2.boundingRect(approx)
            
            custom_image_resized = cv2.imread("300.jpg")
            custom_image_resized = cv2.resize(custom_image_resized, (w, h))
            result = img.copy()
            result[y:y+h, x:x+w] = custom_image_resized
            cv2.imshow('result', result)
           
    cv2.imshow('Object Detection', img)
    cv2.imshow('canny', canny)

    # 退出循环条件：按下"Esc"键
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
