'''
Train PredNet on Generated sequences.
'''

import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils import SequenceGenerator
from agent import Agent
# from border_ownership.ploter import plot_seq_prediction
import matplotlib.pyplot as plt
# from border_ownership.video_straight_reader import VS_reader
from kitti_settings import *

WEIGHTS_DIR = './output_models'

save_model = True # if weights will be saved
train_model = True
#weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_test_no_noise.hdf5')  # where weights will be saved
#json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model_test_no_noise.json')
#weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_test_noise.hdf5')  # where weights will be saved
#json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model_test_noise.json')
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights_simple.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model_simple.json')
#weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + 'prednet_kitti_weights.hdf5')  # where weights will be saved
#json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')

## Training parameters
nb_epoch = 100
batch_size = 2

# Model parameters
n_channels, im_height, im_width = (3, 128, 160)
#n_channels, im_height, im_width = (3, 256, 320)
chs_first = K.image_data_format() == 'channels_first'
input_shape = (n_channels, im_height, im_width) if chs_first else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 4, 8, 16)
R_stack_sizes = stack_sizes
filt_width = 3
A_filt_sizes = (filt_width, filt_width, filt_width)
Ahat_filt_sizes = (filt_width, filt_width, filt_width, filt_width)
R_filt_sizes = (filt_width, filt_width, filt_width, filt_width)
layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
nt = 11  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
time_loss_weights[0] = 0
# stddev = 0.05 # internal noise

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True, 
                #   stddev=stddev
                  )

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(inputs=inputs, outputs=final_errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

## create simple training data set
#vsr = VS_reader()
#video = vsr.read_video_ppd(imshape=(im_height, im_width), add_rgb=True, chs_first=chs_first)
#x_train = video[np.newaxis]

# moving square
x_train = np.ones((batch_size, nt, *input_shape)) * 0.5
for t in range(x_train.shape[1]):
    for vi in range(x_train.shape[0]):
        speed = vi
        xpos, ypos = speed * t, im_height//2
        square_len = 5
        x_train[vi, t, ypos: ypos + square_len, xpos:xpos + square_len, :] = 1

y_train = np.zeros((x_train.shape[0], 1))
x_val = x_train
y_val = np.zeros((x_val.shape[0], 1))

#lr_schedule = lambda epoch: 0.01 if epoch < 75 else 0.001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs

callbacks = [LearningRateScheduler(lr_schedule)]
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

if train_model:
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks, validation_data=(x_val, y_val))

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

# plot the prediction on the validation set
sub = Agent(
    # turn_noise=True
            )
sub.read_from_json(json_file, weights_file)

#### output prediction
prediction = sub.output(x_val, cha_first=False, is_upscaled=False, output_mode='prediction')
print(x_val.shape)
print(prediction.shape)
# for i in range(x_val.shape[0]):
    # plot_seq_prediction(x_val[i], prediction[i])
# plt.show()

#### output neural response
output = sub.output(x_val[0][np.newaxis], cha_first=False, is_upscaled=False, output_mode='R2')
time = np.arange(output.shape[1])
plt.figure()
plt.plot(time, output[0, :, 2, 2, 7])
plt.show()
