import cv2
import numpy as np
import os
from tqdm import tqdm

for root, dirs, files in os.walk("./src/videos", topdown=False):
	path_2_write = root.replace('videos', 'images')
	if not os.path.exists(path_2_write):
		os.mkdir(path_2_write)
		
	for file in tqdm(files):
		namer = 0
		capture = cv2.VideoCapture(os.path.join(root, file))
		
		while True:
			ret, frame = capture.read()

			if not ret:
				break

			# Frame width and height can be resized by fx and fy..
			resizedFrame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
			

			file_name = os.path.join(path_2_write, file)[:-4] + f'{namer}.png'
			cv2.imwrite(file_name, resizedFrame)
			namer += 1

			k = cv2.waitKey(20) & 0xFF
			if k == ord('q'):
				break

capture.release()
cv2.destroyAllWindows()
