# encoding: utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

class HedNetModel(object):
	"""docstring for HedNetModel"""
	def __init__(self, pretrain_model_path):
		super(HedNetModel, self).__init__()
		self.vgg16_params = self.load_vgg_param(pretrain_model_path)
		

	def load_vgg_param(self,path):
		params = np.load(path,encoding='latin1').item()
		return params

	def upsampling(self,bottom,feature_map_size):
		# feature_map_size: int [h,w]
		return tf.image.resize_bilinear(bottom,size=feature_map_size)
        
	def get_conv_filter(self,name):
		init=tf.constant_initializer(self.vgg16_params[name][0])
		shape=self.vgg16_params[name][0].shape
		var=tf.get_variable('weights',shape=shape,dtype=tf.float32,initializer=init)
		return var
        
	def get_bias(self,name):
		init=tf.constant_initializer(self.vgg16_params[name][1])
		shape=self.vgg16_params[name][1].shape # tuple
		bias=tf.get_variable('biases',shape=shape,dtype=tf.float32,initializer=init)
		return bias
        
	def conv_bn_f(self,bottom,is_training,name):
		# finu-tune and batch_norm ; fine-tune not shape,shape had known
		with tf.variable_scope(name) as scope:
			weights=self.get_conv_filter(name)
			biases=self.get_bias(name)
			out=tf.nn.conv2d(bottom,filter=weights,strides=[1,1,1,1],padding='SAME')
			out=tf.nn.bias_add(out,biases)
			#bn before relu and train True test False
			out=tf.contrib.layers.batch_norm(out,center=True,scale=True,is_training=is_training)
			out=tf.nn.relu(out)
		return out
    
	def conv_bn(self,bottom,ksize,is_training,name):
        # initialize and batch_norm ; stride =[1,1,1,1]
		with tf.variable_scope(name) as scope:
			weights=tf.get_variable('weights',ksize,tf.float32,initializer=xavier_initializer())
			biases=tf.get_variable('biases',[ksize[-1]],tf.float32,initializer=tf.constant_initializer(0.0))
			out=tf.nn.conv2d(bottom,filter=weights,strides=[1,1,1,1],padding='SAME')
			out=tf.nn.bias_add(out,biases)
			#bn
			out=tf.contrib.layers.batch_norm(out,center=True,scale=True,is_training=is_training)
			out=tf.nn.relu(out)
		return out
        
	def max_pool(self,bottom,name):
		out=tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)
		return out

	def build(self,input_image,is_training):
		r,g,b=tf.split(input_image,3,axis=3) #这里输出为类型为：shape=(384, 544, 1) dtype=uint8
		b = tf.cast(b,tf.float32)
		g = tf.cast(g,tf.float32)
		r = tf.cast(r,tf.float32)
		image=tf.concat([b*0.00390625,g*0.00390625,r*0.00390625],axis=3) #输出类型为：shape=(384, 544, 3) dtype=float32

		# vgg16
		# block 1
		self.conv1_1=self.conv_bn_f(image,is_training=is_training,name='conv1_1')
		self.conv1_2=self.conv_bn_f(self.conv1_1,is_training=is_training,name='conv1_2')
		self.pool1=self.max_pool(self.conv1_2,name='pool1')
		# block 2
		self.conv2_1=self.conv_bn_f(self.pool1,is_training=is_training,name='conv2_1')
		self.conv2_2=self.conv_bn_f(self.conv2_1,is_training=is_training,name='conv2_2')
		self.pool2=self.max_pool(self.conv2_2,name='pool2')
		# block 3
		self.conv3_1=self.conv_bn_f(self.pool2,is_training=is_training,name='conv3_1')
		self.conv3_2=self.conv_bn_f(self.conv3_1,is_training=is_training,name='conv3_2')
		self.conv3_3=self.conv_bn_f(self.conv3_2,is_training=is_training,name='conv3_3')
		self.pool3=self.max_pool(self.conv3_3,name='pool3')
		# block 4
		self.conv4_1=self.conv_bn_f(self.pool3,is_training=is_training,name='conv4_1')
		self.conv4_2=self.conv_bn_f(self.conv4_1,is_training=is_training,name='conv4_2')
		self.conv4_3=self.conv_bn_f(self.conv4_2,is_training=is_training,name='conv4_3')
		self.pool4=self.max_pool(self.conv4_3,name='pool4')
		# block 5
		self.conv5_1=self.conv_bn_f(self.pool4,is_training=is_training,name='conv5_1')
		self.conv5_2=self.conv_bn_f(self.conv5_1,is_training=is_training,name='conv5_2')
		self.conv5_3=self.conv_bn_f(self.conv5_2,is_training=is_training,name='conv5_3')

		self.upscore_dsn1_1=self.conv_bn(self.conv1_1,ksize=[1,1,64,1],is_training=is_training,name='upscore_dsn1_1')
		self.upscore_dsn1_2=self.conv_bn(self.conv1_2,ksize=[1,1,64,1],is_training=is_training,name='upscore_dsn1_2')
        
		self.score_dsn2_1=self.conv_bn(self.conv2_1,ksize=[1,1,128,1],is_training=is_training,name='score_dsn2_1')
		self.upscore_dsn2_1=self.upsampling(self.score_dsn2_1,tf.shape(image)[1:3])
        
		self.score_dsn2_2=self.conv_bn(self.conv2_2,ksize=[1,1,128,1],is_training=is_training,name='score_dsn2_2')
		self.upscore_dsn2_2=self.upsampling(self.score_dsn2_2,tf.shape(image)[1:3])
        
		self.score_dsn3_1=self.conv_bn(self.conv3_1,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_1')
		self.upscore_dsn3_1=self.upsampling(self.score_dsn3_1,tf.shape(image)[1:3])
        
		self.score_dsn3_2=self.conv_bn(self.conv3_2,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_2')
		self.upscore_dsn3_2=self.upsampling(self.score_dsn3_2,tf.shape(image)[1:3])
        
		self.score_dsn3_3=self.conv_bn(self.conv3_3,ksize=[1,1,256,1],is_training=is_training,name='score_dsn3_3')
		self.upscore_dsn3_3=self.upsampling(self.score_dsn3_3,tf.shape(image)[1:3])
        
		self.score_dsn4_1=self.conv_bn(self.conv4_1,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_1')
		self.upscore_dsn4_1=self.upsampling(self.score_dsn4_1,tf.shape(image)[1:3])
        
		self.score_dsn4_2=self.conv_bn(self.conv4_2,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_2')
		self.upscore_dsn4_2=self.upsampling(self.score_dsn4_2,tf.shape(image)[1:3])
        
		self.score_dsn4_3=self.conv_bn(self.conv4_3,ksize=[1,1,512,1],is_training=is_training,name='score_dsn4_3')
		self.upscore_dsn4_3=self.upsampling(self.score_dsn4_3,tf.shape(image)[1:3])
                
		self.score_dsn5_1=self.conv_bn(self.conv5_1,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_1')
		self.upscore_dsn5_1=self.upsampling(self.score_dsn5_1,tf.shape(image)[1:3])
        
		self.score_dsn5_2=self.conv_bn(self.conv5_2,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_2')
		self.upscore_dsn5_2=self.upsampling(self.score_dsn5_2,tf.shape(image)[1:3])
        
		self.score_dsn5_3=self.conv_bn(self.conv5_3,ksize=[1,1,512,1],is_training=is_training,name='score_dsn5_3')
		self.upscore_dsn5_3=self.upsampling(self.score_dsn5_3,tf.shape(image)[1:3])
        
		self.concat=tf.concat([self.upscore_dsn1_1,self.upscore_dsn1_2,self.upscore_dsn2_1,self.upscore_dsn2_2,self.upscore_dsn3_1,self.upscore_dsn3_2,self.upscore_dsn3_3,self.upscore_dsn4_1,self.upscore_dsn4_2,self.upscore_dsn4_3,self.upscore_dsn5_1,self.upscore_dsn5_2,self.upscore_dsn5_3],axis=3)
        
		self.score=self.conv_bn(self.concat,ksize=[1,1,13,1],is_training=is_training,name='score')
		#self.softmax=tf.nn.softmax(self.score+tf.constant(1e-4))
        
		#self.pred=tf.argmax(self.softmax,axis=-1)
		return self.score


