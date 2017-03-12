from __future__ import print_function
import argparse
import datetime
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import numpy as np
import time

class TrackedObject:
	tracking_points = None
	close_points = None
	last_center = None
	good_features = None
	id = 0
	active = True
	inactive_frames = 0
	tracker = None

	def __init__(self, new_id):
		self.tracking_points = []
		self.close_points = []
		self.good_features = []
		self.last_center = None
		self.id = new_id
		self.active = True
		self.inactive_frames = 0
		self.tracker = cv2.Tracker_create("KCF")
		
#Will be set in main()
min_area = 0
threshold_x = 0
threshold_y = 0
inactive_cutoff = 0
dilate_iterations = 0
thresh_min = 0
kernel = None
fgbg = None

def change_kernel_size(value):
	global kernel
	if (value % 2 == 0):
		value += 1
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(value,value))

def change_threshold_x(value):
	global threshold_x
	threshold_x = value

def change_threshold_y(value):
	global threshold_y
	threshold_y = value

def change_frame_cutoff(value):
	global inactive_cutoff
	inactive_cutoff = value
	
def change_min_area(value):
	global min_area
	min_area = value
	
def change_dilate(value):
	global dilate_iterations
	dilate_iterations = value
	
def change_thresh(value):
	global thresh_min
	thresh_min = value
	
def main():
	global thresh_min, dilate_iterations, thresh_min, min_area, threshold_x, threshold_y, inactive_cutoff, kernel, fgbg
	cv2.namedWindow('Controls')
	cv2.ocl.setUseOpenCL(False)

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video-path", type=str, help="path to video file")
	ap.add_argument("-a", "--min-area", type=int, default=700, help="minimum area size")
	ap.add_argument("-k", "--kernel-size", type=int, default=3, help="kernel size")
	ap.add_argument("-tx", "--threshold-x", type=int, default=40, help="threshold")
	ap.add_argument("-ty", "--threshold-y", type=int, default=70, help="threshold")
	ap.add_argument("-c", "--inactive-cutoff", type=int, default=30, help="inactive cutoff")
	args = vars(ap.parse_args())

	dilate_iterations = 9
	thresh_min = 150
	min_area = args["min_area"]
	threshold_x = args["threshold_x"]
	threshold_y = args["threshold_y"]
	inactive_cutoff = args["inactive_cutoff"]
	
	cv2.createTrackbar('kernel_size', 'Controls', 3, 19, change_kernel_size)
	cv2.createTrackbar('min_area', 'Controls', 700, 2000, change_min_area)
	cv2.createTrackbar('threshold_x', 'Controls', 30, 500, change_threshold_x)
	cv2.createTrackbar('threshold_y', 'Controls', 40, 500, change_threshold_y)
	cv2.createTrackbar('frames_cutoff', 'Controls', 30, 500, change_frame_cutoff)
	cv2.createTrackbar('dilate_itr', 'Controls', 9, 20, change_dilate)
	cv2.createTrackbar('thresh_min', 'Controls', 150, 255, change_thresh)
	
	# capture frames from a video
	camera = cv2.VideoCapture(args["video_path"])
	width = int(camera.get(3))
	height = int(camera.get(4))
	record = cv2.VideoWriter("./output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (width, height))
	#camera = cv2.VideoCapture('./Walking/img/%04d.jpg')

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(args["kernel_size"],args["kernel_size"]))
	fgbg = cv2.createBackgroundSubtractorMOG2(300, 80, True)

	objs = []
	newly_expired = []
	expired_objs = []
	grabbed = True
	id = 0
	frame_id = 0
		
	while grabbed == True:
		frame_id += 1
		contours = []
		# grab the current frame and initialize the occupied/unoccupied
		# text
		(grabbed, frame) = camera.read()
		height, width, channels = frame.shape
		image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		
		thresh = fgbg.apply(image)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
		thresh = cv2.dilate(thresh, None, iterations=dilate_iterations)
		val, thresh = cv2.threshold(thresh, thresh_min, 255, cv2.THRESH_BINARY)
		_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if frame_id > 5:
			for obj in objs:
				obj.close_points = [obj.last_center]
				obj.inactive_frames = obj.inactive_frames + 1
			
			rects = []
			cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
			# loop over the contours
			for c in cnts:
				# if the contour is too small, ignore it
				area = cv2.contourArea(c)
				(x, y, w, h) = cv2.boundingRect(c)
				
				if area < min_area:
					continue
				point = (int((x + w)), int((y + h)))
				if point[0] <= 20 or point[1] <= 20 or point[0] >= (width - 20) or point[1] >= (height - 20):
					continue
					
				rects.append((x, y, w, h))
			
			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			pick = non_max_suppression(rects, probs=None, overlapThresh=0.15)
			for (xA, yA, xB, yB) in pick:
				center_point = (int(xA + ((xB - xA)/2)), int(yA + ((yB - yA)/2)))
				closest = None
				for obj in objs:
					for point in obj.close_points:
						difference = (abs(point[0] - center_point[0]),abs(point[1] - center_point[1]))
						if difference[0] <= threshold_x and difference[1] <= threshold_y:
							if closest is None:
								closest = obj
							else:
								closest_difference = (abs(closest.last_center[0] - center_point[0]),abs(closest.last_center[1] - center_point[1]))
								if difference[0] <= closest_difference[0] and difference[1] <= closest_difference[1]:
									closest = obj
				
				if closest is not None:
					closest.last_center = center_point
					closest.inactive_frames = 0
				else:
					closest = TrackedObject(id)
					closest.tracker.init(frame, (xA, yA, xB - xA, yB - yA))
					id += 1
					closest.last_center = center_point
					objs.append(closest)
				cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
				
			for obj in objs:
				ok, bbox = obj.tracker.update(frame)
				if ok:
					p1 = (int(bbox[0]), int(bbox[1]))
					p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
					#cv2.rectangle(frame, p1, p2, (0,0,255))
					obj.last_center = (int(p1[0] + ((p2[0] - p1[0])/2)), int(p1[1] + ((p2[1] - p1[1])/2)))
 
					obj.tracking_points.append((obj.last_center[0], obj.last_center[1]))
					pts = obj.tracking_points
					if obj.active == True:
						cv2.putText(frame,"ID:" + str(obj.id), (obj.last_center[0],obj.last_center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
					for i in range(1, len(pts)):
						# if either of the tracked points are None, ignore
						# them
						if pts[i - 1] is None or pts[i] is None:
							continue
				 
						# otherwise, compute the thickness of the line and
						# draw the connecting lines
						cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2)				
				else:
					obj.active = False
					newly_expired.append(obj)
					
				if obj.inactive_frames >= inactive_cutoff:
					obj.active = False
					newly_expired.append(obj)
		
		for obj in newly_expired:
			expired_objs.append(obj)
			objs.remove(obj)
			
		newly_expired = []
		 
		# show the output image
		cv2.imshow("Detections", frame)
		cv2.imshow("Thresh", thresh)
		record.write(frame)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break
			
if __name__ == "__main__":
    main()
