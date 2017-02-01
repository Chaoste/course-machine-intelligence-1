import utils
import numpy as np
from ml import CNN
import os

orig_data = dict(train=utils.read('digits/digit_train'), test=utils.read('digits/digit_test'))
data = utils.add_morphed_data(orig_data)
model_dir = os.path.abspath('.').replace('\\', '/') + '/model/'
cnn = CNN(checkpoint_dir=model_dir)
cnn.fit(data['train']['images'], data['train']['labels'], epochs=250, save=True)
