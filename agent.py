# A wrapper to PredNet using better interface
import os
import numpy as np

import tensorflow as tf
from prednet import PredNet
from keras.models import Model, model_from_json
from keras import backend as K
from keras.layers import Input

class Agent():
    def __init__(self, gpu=None):
        '''
        gpu (str or None): specifying the GPU name where this agent will run on. this name should be in an acceptable format for tf.device(). If None, default option will be used
        '''
        self.gpu = gpu
        # self.turn_noise = turn_noise

    def read_from_json(self, json_file, weights_file):
        '''
        json_file (str): typically would be os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
        weights_file (str): typically would be os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')
        '''
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()
        self.json_string = json_string
        self.weights_file = weights_file

    def _build_test_prednet(self, output_mode='prediction'):
        self.train_model = model_from_json(self.json_string, custom_objects = {'PredNet': PredNet}) # this provides default model with output_mode as prediction
        self.train_model.load_weights(self.weights_file)
        # Create testing model with user-defined output_mode
        layer_config = self.train_model.layers[1].get_config()
        layer_config['output_mode'] = output_mode
        self.data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        # layer_config['turn_noise'] = self.turn_noise
        self.test_prednet = PredNet(weights=self.train_model.layers[1].get_weights(), **layer_config)

        return self.train_model, self.test_prednet

    def output_multiple(self, seq_batch, cha_first=None, is_upscaled=True, output_mode=['prediction'], batch_size=10):
        '''
        output multiple modules' neural response
        '''
        feature = {}
        for key in output_mode:
            feature[key] = self.output(seq_batch, cha_first=cha_first, is_upscaled=is_upscaled, output_mode=key, batch_size=batch_size)
            K.clear_session()

        return feature

    def _output(self, seq_batch, cha_first=None, is_upscaled=True, output_mode='prediction', batch_size=10):
        '''
        input:
          seq_batch (np array, numeric, [number of sequences, number of images in each sequence, imshape[0], imshape[1], 3 channels]): if cha_first is false, the final three dimension should be imshape[0], imshape[1], 3 channels; if cha_first is ture, the final three dimensions should be 3 channels, imshape[0], imshape[1].
          is_upscaled (bool): True means the RGB value in the seq ranges from 0 to 255 and need to be normalized. The output seq_hat RGB values are in the same range as the input seq. Note the data processed by SequenceGenerator is from 0 to 1
        '''
        self.train_model, self.test_prednet = self._build_test_prednet(output_mode) # create models

        input_shape = list(self.train_model.layers[0].batch_input_shape[1:]) # find the input shape, (number of images, 3 channels, imshape[0], imshape[1]) if the channel_first = True
        input_shape[0] = seq_batch.shape[1]
        inputs = Input(shape=tuple(input_shape))
        predictions = self.test_prednet(inputs)
        test_model = Model(inputs=inputs, outputs=predictions)

        seq_wrapper = seq_batch
            
        if is_upscaled:
            seq_wrapper = seq_wrapper / 255

        if self.data_format == 'channels_first':
            seq_tran = np.transpose(seq_wrapper, (0, 1, 4, 2, 3)) # make it channel first
            seq_hat = test_model.predict(seq_tran, batch_size=batch_size)
            seq_hat = np.transpose(seq_hat, (0, 1, 3, 4, 2)) # convert to original shape
        else:
            seq_hat = test_model.predict(seq_wrapper, batch_size=batch_size)

        if (is_upscaled) and (output_mode == 'prediction'):
            seq_hat = seq_hat * 255
        K.clear_session()
        return seq_hat

    def output(self, seq_batch, cha_first=None, is_upscaled=True, output_mode='prediction', batch_size=10):
        '''
        this is a wrapper for self._output for specifying self.gpu option
        '''
        if self.gpu is None:
            return self._output(seq_batch, cha_first, is_upscaled, output_mode, batch_size)
        else:
            with tf.device(self.gpu):
                return self._output(seq_batch, cha_first, is_upscaled, output_mode, batch_size)

    def get_config(self):
        self._build_test_prednet()
        return self.test_prednet.get_config()

class Agent_RF_Wraper():
    ''' with proper interface for find rf of the prednet'''
    def __init__(self, sub, num_frames, output_mode='E0', meta_batch_size=None, batch_size=None):
        '''
        sub: an Agent class
        meta_batch_size: split videos to every meta_batch_size for prednet to generate one output. All outputs then will be concatenate
        '''
        self.sub = sub
        self.num_frames = num_frames
        self.output_mode = output_mode
        self.meta_batch_size = meta_batch_size
        self.batch_size = batch_size

    def predict(self, input_image_batch):
        '''
        input:
          input_image_batch (n_images, width, height, chs)
        output:
          output (ndarray (n_images, width_layer, height_leyer, chs_layer)): average response as the neural response to one video (static image video)
        '''

        input_video_batch = np.repeat( input_image_batch[:, np.newaxis], self.num_frames, axis=1 ) # (n_videos, num_frames, width, height, chs) where n_videos is the same as the n_images of input_image_batch

        if self.meta_batch_size is None:
            meta_batch_size = input_video_batch.shape[0]
        else:
            meta_batch_size = self.meta_batch_size
        if self.batch_size is None: 
            batch_size = meta_batch_size
        else:
            batch_size = self.batch_size

        n_video = input_video_batch.shape[0]
        output_list = []
        start_point = np.arange(0, n_video, meta_batch_size)
        stop_point = np.append(start_point[1:], n_video)
        for i in range(start_point.shape[0]):
            print('metabatch', i)

            output = self.sub.output(input_video_batch[start_point[i]:stop_point[i]], output_mode=self.output_mode, batch_size=batch_size, is_upscaled=False) # (n_videos, num_frames, width_layer, height_layer, chs_layer)
            output = np.mean(output, axis=1) # average response as the neural response to one video (static image video)
            output_list.append(output)
        output = np.concatenate(output_list, axis=0)
        return output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import predusion.immaker as immaker
    from kitti_settings import *

    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

    imshape = (128, 160)
    square = immaker.Square(imshape, background=150)
    seq_gener = immaker.Seq_gen()

    im = square.set_full_square(color=[255, 0, 0])
    seq_repeat = seq_gener.repeat(im, 4)[None, ...] # broacast one axis

    sub = Agent()
    sub.read_from_json(json_file, weights_file)

    ##### show the prediction
    seq_pred = sub.output(seq_repeat)
    f, ax = plt.subplots(2, 3, sharey=True, sharex=True)
    for i, sq_p, sq_r in zip(range(3), seq_pred[0], seq_repeat[0]):
        ax[0][i].imshow(sq_r.astype(int))
        ax[1][i].imshow(sq_p.astype(int))

    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.show()

    ##### show the R2 neural activity
    r2 = sub.output(seq_repeat, output_mode='R2') # if output is not prediction, the output shape would be (number of images in a seq, a 3d tensor represent neural activation)
    print(r2.shape)
