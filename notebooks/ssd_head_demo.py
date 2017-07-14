
# coding: utf-8

# In[2]:
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import os
import math
import random
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sys.path.append('../')
from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization


slim = tf.contrib.slim


parser = argparse.ArgumentParser(description = 'Get min Confident lvl to display.')
parser.add_argument('--c', type = float, help='minconfidence needed to display box', default=.2)
parser.add_argument('--s', type = float, help='min select threshold needed to display box', default=.92)
parser.add_argument('--a', type = int, help='minmum_area', default=3000)
args = parser.parse_args()
min_conf = args.c
min_select = args.s
min_area = args.a

# In[5]:



# In[6]:

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# ## SSD 300 Model
# 
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
# 
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# In[7]:

# Input placeholder.
net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
	img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
	predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = '../checkpoints/model.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[8]:

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
	# Run SSD network.
	rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
															  feed_dict={img_input: img})

	# Get classes and bboxes from the net outputs.
	rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
			rpredictions, rlocalisations, ssd_anchors,
			select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
	rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
	rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
	rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
	# Resize bboxes to original image shape. Note: useless for Resize.WARP!
	rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
	return rclasses, rscores, rbboxes

cap = cv2.VideoCapture(0)

def convert_to_rvec(x, y, z):
	x = np.deg2rad(x)
	y = np.deg2rad(x)
	z = np.deg2rad(x)

	x = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
	y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
	z = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

	rvec = np.matmul(np.matmul(z, y), x)

	return rvec


def visualize_box(img, rclasses, rscores, rbboxes):
	cropped_head = []
	for ind, box in enumerate(rbboxes):
		topleft = ( int(box[1]*img.shape[1]), int(box[0]*img.shape[0]))
		botright = (int(box[3]*img.shape[1]), int(box[2]*img.shape[0]))
		area = (botright[0]-topleft[0])*(botright[1]-topleft[1])
		if area > min_area:
			# cropped_head.append(img[topleft[1]:botright[1], topleft[0]:botright[0]])
			resized_head = cv2.resize(img[topleft[1]:botright[1], topleft[0]:botright[0]], (64, 64))

			cv2.rectangle(img, topleft, botright, (0, 255, 0), 2)

			pitch = my_head_pose_estimator.return_pitch(resized_head)  # Evaluate the pitch angle using a CNN
			yaw = my_head_pose_estimator.return_yaw(resized_head)  # Evaluate the yaw angle using a CNN
			roll = my_head_pose_estimator.return_roll(resized_head)

			display_pitch = "pitch: " + str(pitch[0, 0, 0])
			display_yaw = "yaw: " + str(yaw[0, 0, 0])
			display_roll = "roll: " + str(roll[0, 0, 0])

			axis = np.float32([[50, 0, 0], [0, 50, 0], [0, 0, 50]])

			rvec = convert_to_rvec(yaw[0, 0, 0], pitch[0, 0, 0], roll[0, 0, 0])
			camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
			camera_matrix = np.float32([[602.10618226, 0.0, 320.27333589],
											[0.0, 603.55869786, 229.7537026],
											[0.0, 0.0, 1.0]])
			tvec = np.float32([0,0,0])
			imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)
			print(imgpts)
			print(imgpts.shape)
			time.sleep(10)

			cv2.line(img, topleft, tuple(imgpts[1].ravel()), (0, 255, 0), 3)

			cv2.putText(img, display_pitch, (topleft[0], botright[1]-60), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 10), 1, cv2.LINE_AA)
			cv2.putText(img, display_yaw, (topleft[0], botright[1]-40), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 20, 255), 1, cv2.LINE_AA)
			cv2.putText(img, display_roll, (topleft[0], botright[1]-20), cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 255, 255), 1, cv2.LINE_AA)
			cv2.putText(img, str(rscores[ind]),topleft, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
	return img

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('head_demo.avi', fourcc, 20.0, (640, 480))

sess = tf.Session()
my_head_pose_estimator = CnnHeadPoseEstimator(sess)  # Head pose estimation object
# dir_path = os.path.dirname(os.path.realpath(__file__))
pitchfile_path = "/home/walter/Documents/others_git/deepgaze/etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"
yawfile_path = "/home/walter/Documents/others_git/deepgaze/etc/tensorflow/head_pose/yaw/cnn_cccdd_30k"
rollfile_path = "/home/walter/Documents/others_git/deepgaze/etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"

my_head_pose_estimator.load_pitch_variables(pitchfile_path)
my_head_pose_estimator.load_yaw_variables(yawfile_path)
my_head_pose_estimator.load_roll_variables(rollfile_path)


while True:
	if not cap.isOpened():
		print('did not load Cam')
		pass
	ret, frame = cap.read()
	img = frame
	# img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	rclasses, rscores, rbboxes = process_image(img, select_threshold=min_select, nms_threshold=min_conf)
	img = visualize_box(img, rclasses, rscores, rbboxes)
	# out.write(img)

	cv2.imshow('demo', img)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
cap.release()
out.release()
