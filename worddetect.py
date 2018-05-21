# encoding: utf-8
import tensorflow as tf
import csv 
import numpy as np

train_image_path = "train_data/image_data/"
train_label_path = "train_data/label_data/"

def read_image_file(imagepath):
	file_content = tf.read_file(imagepath)
	image_content = tf.image.decode_jpeg(file_content,channels=3)
	with tf.Session() as sess:
		image = sess.run(image_content)
		return image

def read_label_file(labelpath):
	file_content = tf.read_file(labelpath)
	image_content = tf.image.decode_png(file_content,channels=1)
	with tf.Session() as sess:
		image = sess.run(image_content)
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

		
class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, images,labels):
		super(Dataset, self).__init__()
		self._images = images
		self._labels = labels
		self._num_examples = images.shape[0]
		self._index_epochs = 0
		self._epochs_completed = 0

	@property
	def images(self):
		return self._images
	@property
	def labels(self):
		return self._labels
	@property
	def num_examples(self):
		return self._num_examples
	@property
	def index_epochs(self):
		return self._index_epochs
	@property
	def epochs_completed(self):
		return self._epochs_completed
	
		
	def next_batch(self,batch_size,shuffle=True):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_epochs
		# Shuffle for the first epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self._num_examples)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]
		else:
			pass

		# Go to the next epoch
		if start + batch_size > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			images_rest_part = self._images[start:self._num_examples]
			labels_rest_part = self._labels[start:self._num_examples]

			# Shuffle the data
			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._images = self.images[perm]
				self._labels = self.labels[perm]
			else:
				pass

			# Start next epoch
			start = 0
			self._index_epochs = batch_size - rest_num_examples
			end = self._index_epochs
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)

		else:
			self._index_epochs += batch_size
			end = self._index_epochs
			return self._images[start:end],self._labels[start:end]

class DataModel(object):
	"""docstring for DataModel"""
	def __init__(self, imageData,labelData,imagepath,labelpath):
		super(DataModel, self).__init__()
		self._imageData = imageData
		self._labelData = labelData
		self._imagepath = imagepath
		self._labelpath = labelpath 

	def discription(self):
		print "imagepath" + " : " +self._imagepath
		print "labelpath" + " : " +self._labelpath
		print "imageData" + " : " 
		print self._imageData
		print "labelData" + " : "
		print self._labelData

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

class DataModelSet(object):
	"""docstring for ClassName"""
	def __init__(self, dataModel):
		super(DataModelSet, self).__init__()
		self._dataModel = dataModel
		self._num_examples = dataModel.shape[0]
		self._index_epochs = 0
		self._epochs_completed = 0

	@property
	def dataModel(self):
		return self._dataModel
	@property
	def num_examples(self):
		return self._num_examples
	@property
	def index_epochs(self):
		return self._index_epochs
	@property
	def epochs_completed(self):
		return self._epochs_completed
	def next_batch(self,batch_size,shuffle=True):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_epochs
		# Shuffle for the first epoch
		if self._epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self._num_examples)
			np.random.shuffle(perm0)
			self._dataModel = self.dataModel[perm0]
		else:
			pass

		# Go to the next epoch
		if start + batch_size > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			rest_num_examples = self._num_examples - start
			dataModel_rest_part = self._dataModel[start:self._num_examples]

			# Shuffle the data
			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._dataModel = self.dataModel[perm]
			else:
				pass

			# Start next epoch
			start = 0
			self._index_epochs = batch_size - rest_num_examples
			end = self._index_epochs
			dataModel_new_part = self._dataModel[start:end]
			return np.concatenate(dataModel_rest_part, dataModel_new_part)

		else:
			self._index_epochs += batch_size
			end = self._index_epochs
			return self._dataModel[start:end]

print "start run..."
filename = "train_data/image_label.csv"
#image_result,label_result = read_image_label_file(filename)

#dataset = Dataset(image_result,label_result)

#image_result_part,label_result_part = dataset.next_batch(10)

#print image_result_part
#print label_result_part


dataModelSet = DataModelSet(read_image_label_model_file(filename))
dataModelpart = dataModelSet.next_batch(5)
for dataModel in dataModelpart:
	dataModel.discription()


print "stop run..."


def hed_net(inputs, batch_size):
    # ref https://github.com/s9xie/hed/blob/master/examples/hed/train_val.prototxt
    with tf.variable_scope('hed', 'hed', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            # vgg16 conv && max_pool layers
            net = slim.repeat(inputs, 2, slim.conv2d, 12, [3, 3], scope='conv1')
            dsn1 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 24, [3, 3], scope='conv2')
            dsn2 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.repeat(net, 3, slim.conv2d, 48, [3, 3], scope='conv3')
            dsn3 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.repeat(net, 3, slim.conv2d, 96, [3, 3], scope='conv4')
            dsn4 = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.repeat(net, 3, slim.conv2d, 192, [3, 3], scope='conv5')
            dsn5 = net
            # net = slim.max_pool2d(net, [2, 2], scope='pool5') # no need this pool layer

            # dsn layers
            dsn1 = slim.conv2d(dsn1, 1, [1, 1], scope='dsn1')
            # no need deconv for dsn1

            dsn2 = slim.conv2d(dsn2, 1, [1, 1], scope='dsn2')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn2 = deconv_mobile_version(dsn2, 2, deconv_shape) # deconv_mobile_version can work on mobile

            dsn3 = slim.conv2d(dsn3, 1, [1, 1], scope='dsn3')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn3 = deconv_mobile_version(dsn3, 4, deconv_shape)

            dsn4 = slim.conv2d(dsn4, 1, [1, 1], scope='dsn4')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn4 = deconv_mobile_version(dsn4, 8, deconv_shape)

            dsn5 = slim.conv2d(dsn5, 1, [1, 1], scope='dsn5')
            deconv_shape = tf.pack([batch_size, const.image_height, const.image_width, 1])
            dsn5 = deconv_mobile_version(dsn5, 16, deconv_shape)

            # dsn fuse
            dsn_fuse = tf.concat(3, [dsn1, dsn2, dsn3, dsn4, dsn5])
            dsn_fuse = tf.reshape(dsn_fuse, [batch_size, const.image_height, const.image_width, 5]) #without this, will get error: ValueError: Number of in_channels must be known.

            dsn_fuse = slim.conv2d(dsn_fuse, 1, [1, 1], scope='dsn_fuse')

    return dsn_fuse, dsn1, dsn2, dsn3, dsn4, dsn5