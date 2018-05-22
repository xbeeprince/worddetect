
import tensorflow as tf
import csv 
import numpy as np
from DataModel import DataModel


train_image_path = "train_data/image_data/"
train_label_path = "train_data/label_data/"
IMAGE_SIZE_WIDTH = 224
IMAGE_SIZE_HEIGHT = 224

def read_image_file(imagepath):
	file_content = tf.read_file(imagepath)
	image_content = tf.image.decode_jpeg(file_content,channels=3)
	#with tf.Session() as sess:
	#	image = sess.run(image_content)
	#	return image

	resize_image = tf.image.resize_images(image_content,[224,224],method = np.random.randint(4))
	with tf.Session() as sess:
		image = sess.run(resize_image)
		return image

def read_label_file(labelpath):
	file_content = tf.read_file(labelpath)
	image_content = tf.image.decode_png(file_content,channels=1)
	#with tf.Session() as sess:
	#	image = sess.run(image_content)
	#	return image

	resize_image = tf.image.resize_images(image_content,[224,224],method = np.random.randint(4))
	with tf.Session() as sess:
		image = sess.run(resize_image)
		return image

def read_image_label_file(csvfilename):
	with open(csvfilename, 'r+') as csv_file:
		reader = csv.reader(csv_file)
		image_result=[]
		label_result=[]
		for row in reader:
			imagepath = train_image_path + row[0]
			labelpath = train_label_path + row[1]
			image_data = read_image_file(imagepath)
			label_data = read_label_file(labelpath)
			image_result.append(image_data)
			print "loading image [" + imagepath + " ]...\r\n"
			label_result.append(label_data)
			print "loading label [" + labelpath + " ]...\r\n"
		return np.array(image_result),np.array(label_result)


def read_image_label_model_file(csvfilename):
	with open(csvfilename, 'r+') as csv_file:
		reader = csv.reader(csv_file)
		model_result=[]
		for row in reader:
			imagepath = train_image_path + row[0]
			labelpath = train_label_path + row[1]
			image_data = read_image_file(imagepath)
			label_data = read_label_file(labelpath)
			dataModel = DataModel(image_data,label_data,imagepath,labelpath)
			#dataModel.discription()
			model_result.append(dataModel)
		return np.array(model_result)