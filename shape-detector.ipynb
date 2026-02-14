import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_path = "/kaggle/input/datasets/nazaninashrafi/img-testt/7242.webp"
image = cv.imread(image_path)

image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
contrast = cv.convertScaleAbs(gray, alpha= 1.6, beta=0)
blur = cv.GaussianBlur(contrast,(11,11),0)
_, binary = cv.threshold(blur, 127, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(
    binary,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE
)

img = image_rgb.copy()

def get_approx(cnt):
    perimeter = cv.arcLength(cnt, True)
    epsilon = 0.04 * perimeter
    approx = cv.approxPolyDP(cnt, epsilon, True)
    return approx

def drawing_contours(img,approx,color):
    cv.drawContours(
        img,
        [approx],
        -1,
        color,
        2
    )

def get_title(text,x,y,color):
    cv.putText(
        img, 
        text, 
        (x, y - 20), 
        cv.FONT_HERSHEY_SIMPLEX, 
        0.8,             
        color,     
        3               
    )

for cnt in contours:
    area = cv.contourArea(cnt)
    if area < 500:
        continue

    approx = get_approx(cnt)
    num_corners = len(approx)
    x, y, w, h = cv.boundingRect(cnt)

    if num_corners == 3:
        # print(f"Corners:{num_corners}")
        # print("triangle")
        drawing_contours(img,approx,(255,255,255))
        get_title("triangle",x,y,(255,255,255))
    elif num_corners == 4:
        # print(f"Corners:{num_corners}")
        # print("rectangle")
        drawing_contours(img,approx,(234, 1, 133))
        get_title("rectangle",x,y,(234, 1, 133))
    else:
        # print(f"Corners:{num_corners}")
        # print("polygon")
        drawing_contours(img,approx,(247, 230, 0))
        get_title("polygon",x,y,(247, 230, 0))

    
plt.imshow(img)
plt.axis("off")
plt.show()
