import numpy as np

tensor = np.ones((2, 10, 128*2, 160*2, 3))

from agent import Agent

sub = Agent()
sub.read_from_json('./model_data_keras2/prednet_kitti_model.json', './model_data_keras2/tensorflow_weights/prednet_kitti_weights.hdf5')

output = sub.output(tensor, output_mode='prediction', batch_size=2, is_upscaled=False)

print(output.shape)

import matplotlib.pyplot as plt

for i in range(10):
    plt.imshow(output[0, i])
    plt.show()
