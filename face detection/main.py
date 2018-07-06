import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haar_files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar_files/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haar_files/haarcascade_smile.xml')


if((raw_input('Cam or Photo?\n')).lower() == 'cam'):
	cap = cv2.VideoCapture(0)
	which = 'cam'

else:
	which = 'image'


while True:
	if(which == 'cam'):
		ret, img = cap.read()

	else:
		img = cv2.imread('images/kimi.jpg')
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.2, 3)

	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 2)

		smile = smile_cascade.detectMultiScale(roi_gray, 1.9, 25)

		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

		for (sx, sy, sw, sh) in smile:
			cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)


	cv2.imshow('image', img)


	k = cv2.waitKey(30) & 0xff

	if k == 27:
		break


try:
	cap.release()

except:
	pass
	
	
cv2.destroyAllWindows()