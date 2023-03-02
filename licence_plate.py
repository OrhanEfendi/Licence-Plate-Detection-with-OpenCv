import cv2
import numpy as np
import pytesseract
import imutils

img_car = cv2.imread("W:/Users/Desktop/pythonProject1/Tessacaret-ORC/image/dataset-card.jpg")

gray_car = cv2.cvtColor(img_car, cv2.COLOR_BGR2GRAY)

filter_car = cv2.bilateralFilter(gray_car, 6, 150, 250)
edged = cv2.Canny(filter_car, 20, 200)

contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(contours)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screen = None

for c in cnts:
    epsilon = 0.018*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    if(len(approx)==4):
        screen = approx
        break
print(screen)
mask = np.zeros(gray_car.shape, np.uint8)
new_img = cv2.drawContours(mask, [screen], 0,(255,255,255), -1)
new_image = cv2.bitwise_and(img_car, img_car, mask=mask)
(x,y) =np.where(mask ==255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray_car[topx:bottomx+1, topy:bottomy+1]
print(x,y)
print(cropped)
text = pytesseract.image_to_string(new_image, lang="eng")
print("Araba_plakasi:",text)
cv2.imshow("Image", img_car)
cv2.imshow("Gray", gray_car)
cv2.imshow("Filter", filter_car)
cv2.imshow("edged", edged)
cv2.imshow("Mask", mask)
cv2.imshow("Plate", new_image)
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
