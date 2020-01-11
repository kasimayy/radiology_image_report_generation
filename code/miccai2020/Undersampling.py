'''
From GitHub: gavinlive/perception
'''
import tensorflow as tf
import os, sys
from . import printt

class Undersampling(object):
    def __init__(self, height,width,  no_central_lines=4, std_frac=0.4, acc_factor=None):
        '''
        Arguments:
        (1) height
        (2) width
        (3) no_central_lines (default: 4)
        (4) std_frac (default: 0.4)
        '''
        self.height = height
        self.width = width
        if(no_central_lines % 2 != 0):
            raise ValueError('expected no_central_lines to be an even integer')

        if(self.height % 2 != 0):
            raise ValueError('height specified is odd. Only even-numbered height is supported here currently.')


        heights = range(0, self.height)
        middle = self.height/2.
        std = float(middle)*std_frac
        self.dist = tf.compat.v1.distributions.Normal(loc=middle, scale=std)
        self.middle=int(middle)
        self.no_central_lines = no_central_lines

        self.use_data_provided_mask = False
        self.mask = None # only for data_provided_mask
        self.visual_mask = None # only for data_provided_mask
        self.acc_factor = self.get_undersample_rate(acc_factor)
    def get_undersample_rate(self, acc_factor):
        '''
        Due to the way that the while loop works for generating the mask,
        we must provide the generate_mask method with a factor that is slightly
        more aggressive than the true factor we require since the algorithm will
        always acquire a line such that it tips the current factor over (greater
        than) the specified factor
        '''
        ur_true = 1./acc_factor
        no_lines = int(self.height * ur_true) # rounds down integer

        ur_true = no_lines/self.height
        ur_overaggressive = (no_lines-1)/self.height
        ur = (ur_true + ur_overaggressive)/2.
        return ur
    def set_mask(self, masks, fftshift=True):
        '''
        We assume that the mask provided has the centre of k-space in
        the centre of the image. If not true, please set the fftshift
        option to False
        '''
        self.use_data_provided_mask = True
        if fftshift is False:
            self.mask = masks
            self.visual_mask = self.fftshift(masks)
        else:
            self.mask = self.fftshift(masks)
            self.visual_mask = masks
    def generate_mask(self, factor=0.016):
        '''
        Generates a 2D Cartesian VD mask with dimensions [self.height, self.width]
        '''
        def _body(current_factor, mask):
            val = tf.cast(self.dist.sample([1]), dtype=tf.int32)
            this_mask = tf.cast(tf.one_hot(val, self.height, axis=0), dtype=tf.int32)
            new_mask = tf.add(mask, tf.reduce_sum(input_tensor=this_mask, axis=1))
            mask = tf.cast(tf.cast(new_mask, dtype=tf.bool), dtype=tf.int32)
            current_factor = tf.divide(tf.cast(tf.reduce_sum(input_tensor=mask), dtype=tf.float32), self.height)
            return current_factor, mask
        cen = int(self.no_central_lines/2)
        mask_vals_1 = tf.range(self.middle, self.middle+cen-1 + 1, dtype=tf.int32)
        mask_vals_2 = tf.range(self.middle-1 - (cen-1), self.middle-1 + 1, dtype=tf.int32)
        mask_vals = tf.concat([mask_vals_1, mask_vals_2], axis=0)
        mask = tf.cast(tf.cast(tf.reduce_sum(input_tensor=tf.one_hot(mask_vals, self.height, axis=0), axis=1), dtype=tf.bool), dtype=tf.int32)
        current_factor = float(self.no_central_lines)/float(self.height)
        current_factor, mask = tf.while_loop(cond=lambda current_factor, mask: current_factor < factor, body=_body, loop_vars=(current_factor, mask))
        #mask = tf.one_hot(mask_vals, self.height, axis=0
        #mask = tf.cast(tf.cast(tf.reduce_sum(tf.one_hot(mask_vals, self.height, axis=0), axis=1), dtype=tf.bool), dtype=tf.int32)
        #print(mask_vals)
        #print(mask.get_shape().as_list())
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1,self.width])
        #print(mask.get_shape().as_list())
        return mask
    def __call__(self, input_images, acc_factor=None, set_seed=False, reset_seed=False):
        '''
        Assumes that you provide a cine [Batch, Nt, H, W]
        Returns:
        (1) Undersampled image (complex)
        (2) Masks for direct use in the TF framework
        (3) Masks that are not useful in the code, but used for display
            since with these masks, the centre of the image corresponds
            to the centre of the k-space
        '''
        if acc_factor is None:
            if not((hasattr(self, "acc_factor") is True) and (self.acc_factor is not None)) is True:
                printt("Acceleration factor not set", error=True, stop=True)
            else:
                acc_factor = self.acc_factor
        else:
            acc_factor = self.get_undersample_rate(acc_factor)


        input_images = tf.cast(input_images, tf.complex64)
        # Returns input_images, masks, visual_masks
        if set_seed is True:
            tf.random.set_seed(1114)
        elif isinstance(set_seed, int) is True:
            tf.random.set_seed(set_seed)
        its = input_images.get_shape().as_list()
        Nt = its[1]
        M = its[0]
        if self.use_data_provided_mask is False:
            masks = []
            for j in range(M):
                mask = []
                for i in range(Nt):
                    mask.append(tf.expand_dims(self.generate_mask(factor=acc_factor), axis=2))
                mask = tf.concat(mask, axis=2)
                mask = tf.transpose(a=mask, perm=[2, 0, 1])
                masks.append(tf.expand_dims(mask, axis=0))
            masks = tf.concat(masks, axis=0) # [M, C, 256, 32]
            visual_mask = masks


            apply_mask = self.fftshift(masks)
        else:
            apply_mask = self.mask
            masks = self.visual_mask
            visual_mask = self.visual_mask

        input_images_k_space = tf.signal.fft2d(input_images) # [M, C, 256, 256]
        apply_mask_2 = tf.cast(apply_mask, tf.complex64)
        output_images_k_space = tf.multiply(input_images_k_space, apply_mask_2)
        output = tf.signal.ifft2d(output_images_k_space) # [M, C, 256, 256]
        return output, apply_mask, visual_mask
    def fftshift(self, input):
        '''
        Assumes that you provide a cine [Batch, Nt, H, W]
        Performs along the "H" dimension (axis=2)
        '''
        its = input.get_shape().as_list()
        first_set = tf.slice(input, [0,0,0,0], [its[0], its[1], self.middle, its[3]])
        second_set = tf.slice(input, [0,0,self.middle,0], [its[0], its[1], self.middle, its[3]])
        return tf.concat([second_set, first_set], axis=2)
