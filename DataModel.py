# encoding: utf-8
import numpy as np
class DataModel(object):
	"""docstring for DataModel"""
	def __init__(self, imageData,labelData,imagepath,labelpath):
		super(DataModel, self).__init__()
		self._imageData = imageData
		threshold, upper, lower = 0, 1, 0
		y = np.where(labelData>threshold, upper, lower) #二值化
		self._labelData = y
		self._imagepath = imagepath
		self._labelpath = labelpath

	@property
	def imageData(self):
		return self._imageData
	@property
	def labelData(self):
	 	return self._labelData
	@property
	def imagepath(self):
	 	return self._imagepath
	@property
	def labelpath(self):
	 	return self._labelpath
	 
	def discription(self):
		print "imagepath" + " : " +self._imagepath
		print "labelpath" + " : " +self._labelpath
		print "imageData" + " : " 
		print self._imageData
		print "labelData" + " : "
		print self._labelData