

# timestamps
import time
#print(time.strftime('%a %H:%M:%S'))

#import datetime
#print('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
#print('Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now()))

class args:
	def __init__(self, data):
		self.data = data
		self.output = 'output'
		self.model = ''
		self.batch_size = 16
		self.num_epochs = 500
		self.optimizer = 'adam'
		self.deconv_layers = 6
		#self.kernels_per_layer = None #[128, 128, 96, 96, 32, 32, 16] #None
		self.kernels_per_layer = [128, 128, 96, 96, 32*2, 32*2, 16*2]
		self.visualize = False
		self.use_yalefaces = False
		self.verbose = True

		self.setModelName()

	def setModelName(self, trainNewModel=True):
		m = "./models/FaceGen.{}.b{}.e{}.d{}.o-{}.h5".format( \
			time.strftime('%a-%H-%M-%S'), self.batch_size, \
			self.num_epochs, self.deconv_layers, self.optimizer)
		if trainNewModel:
			self.output = m
			self.model = None
		else:
			self.model = m
			self.output = 'output'

if __name__ == '__main__':

	# see all GPUs
	#from tensorflow.python.client import device_lib
	#print device_lib.list_local_devices()

	# select GPU
	import os
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
	#os.environ["CUDA_VISIBLE_DEVICES"]="1" # on the mobo
	os.environ["CUDA_VISIBLE_DEVICES"]="0" # in the external slot


	args = args(data="../DB/rafd2-frontal/")

	import faces.train

	faces.train.train_model(args.data, args.output, args.model,
	        batch_size            = args.batch_size,
	        num_epochs            = args.num_epochs,
	        optimizer             = args.optimizer,
	        deconv_layers         = args.deconv_layers,
	        kernels_per_layer     = args.kernels_per_layer,
	        generate_intermediate = args.visualize,
	        use_yale              = args.use_yalefaces,
	        verbose               = True,
	    )
