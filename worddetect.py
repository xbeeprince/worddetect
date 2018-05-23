# encoding: utf-8
import tensorflow as tf
import numpy as np
import DataModel
from DataModelSet import DataModelSet
import ImageIO
from HedNetModel import HedNetModel


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    This is more numerically stable than class_balanced_cross_entropy

    :param logits: size: the logits.
    :param label: size: the ground truth in {0,1}, of the same shape as logits.
    :returns: a scalar. class-balanced cross entropy loss
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y) # the number of 0 in y
    count_pos = tf.reduce_sum(y) # the number of 1 in y (less than count_neg)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)
    return cost


def worddetect():
	print "start run..."
	filename = "train_data/image_label.csv"

	dataModelSet = DataModelSet(ImageIO.read_image_label_model_file(filename))
	print "num_examples : %d" %(dataModelSet.num_examples)

	#dataModelpart = dataModelSet.next_batch(5)
	#for dataModel in dataModelpart:
	#	dataModel.discription()

	epochs = 10000

	# 输入变量,x为图像,y为标签
	x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224,3], name='x')
	y = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224,1], name='y')
	hedmodel = HedNetModel("vgg16.npy")
	score = hedmodel.build(x,is_training = True)
	print "score.shape : "
	print score.shape
	cost = class_balanced_sigmoid_cross_entropy(score,y)
	accucy = 1 - cost;

	# 建立会话
	with tf.Session() as sess:
	# 初始化变量
		sess.run(tf.global_variables_initializer())
		for i in range(epochs):
			print "training... "
			# 批量数据,大小为5
			tmpDataList = []
			tmpLabelList = []
			dataModelpart = dataModelSet.next_batch(5)
			for tmpDataModel in dataModelpart:
				tmpDataList.append(tmpDataModel.imageData)
				tmpLabelList.append(tmpDataModel.labelData)
			x_batch = np.array(tmpDataList)
			y_batch = np.array(tmpLabelList)

			c = sess.run(cost, feed_dict={x:x_batch, y:y_batch})
			print "cost:"
			print c

	print "stop run..."	



if __name__ == '__main__':  
	worddetect()
