# encoding: utf-8

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