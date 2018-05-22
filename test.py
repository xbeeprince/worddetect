# encoding: utf-8
import tensorflow as tf
import numpy as np
import DataModel
from DataModelSet import DataModelSet
import ImageIO
from HedNetModel import HedNetModel




print "start run..."
filename = "train_data/image_label.csv"

dataModelSet = DataModelSet(ImageIO.read_image_label_model_file(filename))
print "num_examples : %d" %(dataModelSet.num_examples)

for x in xrange(1,10):
	dataModelpart = dataModelSet.next_batch(5)
	for dataModel in dataModelpart:
		dataModel.discription()



print "stop run..."