
# encoding: utf-8

from DataModel import DataModel
import numpy as np

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
			return np.append(dataModel_rest_part,dataModel_new_part)
			#return np.concatenate(dataModel_rest_part, dataModel_new_part)

		else:
			self._index_epochs += batch_size
			end = self._index_epochs
			return self._dataModel[start:end]