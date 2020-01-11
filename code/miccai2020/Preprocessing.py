from Undersampling import Undersampling
import numpy as np
import tensorflow_addons as tfa

class MRIScanner(Undersampling):
    def __call__(self, input_images, acc_factor=None, set_seed=False, reset_seed=False):
        '''
        input_images: [B, H, W]
        '''
        input_images = self.preprocess(input_images)
        input_images = self.augment(input_images)
        # Note: 
        super().__call__(input_images, acc_factor=acc_factor, set_seed=set_seed, reset_seed=reset_seed)
    def preprocess(self, input_images):
        '''
        input_images: [B, H, W]
        '''
        this_width = input_images.shape[2]
        this_height = input_images.shape[1]
        new_width = this_width if self.width < this_width else self.width
        new_height = this_height if self.height < this_height else self.height

        pad = None
        pad_width = [0,0]
        pad_height = [0,0]
        if new_width < self.width:
            pad_ = self.width - new_width
            pad_left = np.float(pad_)/2.
            pad_right = pad_ - pad_left
            pad_width = [pad_left, pad_right]
            pad = True
        if new_height < self.height:
            pad_ = self.height - new_height
            pad_left = np.float(pad_)/2.
            pad_right = pad_ - pad_left
            pad_height = [pad_left, pad_right]
            pad = True
        if pad is not None:
            input_images = tf.pad(input_images, [[0,0], pad_width, pad_height])
        return input_images

    def augment_data(self, input_images):
        input_images = tf.expand_dims(input_images, axis=3)
        angles = tf.random.uniform([input_images.shape[0]], minval=-np.pi/4, maxval=np.pi/4)
        input_images = tfa.image.rotate(input_images, angles)
        return input_images