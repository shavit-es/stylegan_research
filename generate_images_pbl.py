# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import random
import tensorflow as tf


def main():
    # Initialize TensorFlow.
    tflib.init_tf()
    with tf.device("/gpu:1"):
        model_name='child-network-snapshot-004705.pkl' #Model 불러오기
        with open(model_name,'rb') as f1:
            _G1, _D1, Gs = pickle.load(f1)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        # Print network details.
        Gs.print_layers()
        image_num = 8 #생성할 이미지 개수
        seed = random.sample(range(0,10000), image_num) #seed 랜덤 샘플링

        for i in range(image_num):
            # Pick latent vector.
            rnd = np.random.RandomState(seed[i])
            latents = rnd.randn(1, Gs.input_shape[1])
            # Generate image.
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            images = Gs.run(latents, None, truncation_psi=1.0, randomize_noise=True, output_transform=fmt)
            # Save image.
            os.makedirs(config.result_dir+'/'+model_name+'/', exist_ok=True)
            png_filename = os.path.join(config.result_dir+'/'+model_name+'/', str(seed[i])+'.png')
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
