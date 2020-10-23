# Code for
# Unsupervised Adversarial Depth Estimation using Cycled Generative Networks
# Andrea Pilzer, Dan Xu, Mihai Puscas, Elisa Ricci, Nicu Sebe
#
# 3DV 2018 Conference, Verona, Italy
#
# parts of the code from https://github.com/mrharicot/monodepth
#

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *
from module import *

monodepth_parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'lr_loss_weight, '
                        'alpha_image_loss,'
                        'full_summary, '
                        'num_gpus')

class stereo_depthGAN_Model(object):
    """unsupervised stereo depthGAN model"""

    def __init__(self, params, branch, mode, left, right, reuse_variables=None, model_index=0):
        self.params = params
        self.branch = branch
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.batch_size = params.batch_size/params.num_gpus
        self.size_W = params.width
        self.size_H = params.height
        self.input_c_dim = 3
        self.output_c_dim = 1
        self.identity = tf.Variable(tf.ones([self.batch_size, self.size_H, self.size_W, 1]), trainable=False)

        self.reuse_variables =None
        self.reuse = None
        self.discriminator = self.discr
        #self.encoder = self.build_vgg
        #self.decoder = self.build_resnet50_dec
        self.fusion = self.fusion_func

        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((params.batch_size, params.height,
                                      64, 64, self.output_c_dim,
                                      mode == 'train'))

        self.build()
        #self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            #self.disp_out = self.disp_output()
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def upsample_nn(self, x, ratio):
        s = x.get_shape().as_list()
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def get_disp_original(self, x):
        disp = self.conv(x, 2, 3, 1, tf.nn.sigmoid)#0.3
        return disp

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disp(self, x):
        disp = self.conv(x, 1, 3, 1, tf.nn.sigmoid)#0.3
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def build_vgg(self,resue=False):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input,  32, 7) # H/2
            conv2 = self.conv_block(conv1,             64, 5) # H/4
            conv3 = self.conv_block(conv2,            128, 3) # H/8
            conv4 = self.conv_block(conv3,            256, 3) # H/16
            conv5 = self.conv_block(conv4,            512, 3) # H/32
            conv6 = self.conv_block(conv5,            512, 3) # H/64
            conv7 = self.conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

            with tf.variable_scope('encoder2'):
                Nconv1 = self.conv_block(self.model_input,  32, 7) # H/2
                Nconv2 = self.conv_block(Nconv1,             64, 5) # H/4
                Nconv3 = self.conv_block(Nconv2,            128, 3) # H/8
                Nconv4 = self.conv_block(Nconv3,            256, 3) # H/16
                Nconv5 = self.conv_block(Nconv4,            512, 3) # H/32
                Nconv6 = self.conv_block(Nconv5,            512, 3) # H/64
                Nconv7 = self.conv_block(Nconv6,            512, 3) # H/128

            with tf.variable_scope('skips2'):
                Nskip1 = Nconv1
                Nskip2 = Nconv2
                Nskip3 = Nconv3
                Nskip4 = Nconv4
                Nskip5 = Nconv5
                Nskip6 = Nconv6

            with tf.variable_scope('decoder2'):
                Nupconv7 = upconv(Nconv7,  512, 3, 2) #H/64
                Nconcat7 = tf.concat([Nupconv7, Nskip6], 3)
                Niconv7  = conv(Nconcat7,  512, 3, 1)

                Nupconv6 = upconv(Niconv7, 512, 3, 2) #H/32
                Nconcat6 = tf.concat([Nupconv6, Nskip5], 3)
                Niconv6  = conv(Nconcat6,  512, 3, 1)

                Nupconv5 = upconv(Niconv6, 256, 3, 2) #H/16
                Nconcat5 = tf.concat([Nupconv5, Nskip4], 3)
                Niconv5  = conv(Nconcat5,  256, 3, 1)

                Nupconv4 = upconv(Niconv5, 128, 3, 2) #H/8
                Nconcat4 = tf.concat([Nupconv4, Nskip3], 3)
                Niconv4  = conv(Nconcat4,  128, 3, 1)
                self.Ndisp4 = self.get_disp(Niconv4)
                Nudisp4  = self.upsample_nn(self.Ndisp4, 2)

                Nupconv3 = upconv(Niconv4,  64, 3, 2) #H/4
                Nconcat3 = tf.concat([Nupconv3, Nskip2, Nudisp4], 3)
                Niconv3  = conv(Nconcat3,   64, 3, 1)
                self.Ndisp3 = self.get_disp(Niconv3)
                Nudisp3  = self.upsample_nn(self.Ndisp3, 2)

                Nupconv2 = upconv(Niconv3,  32, 3, 2) #H/2
                Nconcat2 = tf.concat([Nupconv2, Nskip1, Nudisp3], 3)
                Niconv2  = conv(Nconcat2,   32, 3, 1)
                self.Ndisp2 = self.get_disp(Niconv2)
                Nudisp2  = self.upsample_nn(self.Ndisp2, 2)

                Nupconv1 = upconv(Niconv2,  16, 3, 2) #H
                Nconcat1 = tf.concat([Nupconv1, Nudisp2], 3)
                Niconv1  = conv(Nconcat1,   16, 3, 1)
                self.Ndisp1 = self.get_disp(Niconv1)

    def build_resnet50(self):
        #set convenience functions
        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5,   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def discr(self,image, options, reuse=False, name="discriminator"):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            elif reuse == False and self.reuse == True:
                tf.get_variable_scope().reuse_variables()
            elif reuse == True and self.reuse == False:
                tf.get_variable_scope().reuse_variables()
            elif reuse == False and self.reuse == False:
                assert tf.get_variable_scope().reuse is False

            h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(instance_norm(conv2d(h3, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def fusion_func(self, net_input1, net_input2, name, reuse=False):
        with tf.variable_scope(name):
            conv = self.conv
            if reuse:
                tf.get_variable_scope().reuse_variables()
            input_fusion = tf.concat([net_input1, net_input2], 3)
            conv_fusion = conv(input_fusion, 1, 1, 1, tf.nn.relu)
            return conv_fusion

    def build(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=True):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)
                self.right_pyramid = self.scale_pyramid(self.right, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                if self.params.do_stereo and self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)
                    self.model_input = tf.concat([self.left, self.right],3)
                else:
                    self.model_input = self.left
                    self.model_input2=self.right
                    #return
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None
    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):

             #self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
             #self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]  # for expanding a 1 dimension at end
             #self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]



            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.Ndisp_est  = [self.Ndisp1, self.Ndisp2, self.Ndisp3, self.Ndisp4]

            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]  # for expanding a 1 dimension at end
            self.disp_right_est = [tf.expand_dims(d[:,:,:,0], 3) for d in self.Ndisp_est]

        if self.mode == 'test':
            #self.Ndisp_est  = [self.Ndisp1, self.Ndisp2, self.Ndisp3, self.Ndisp4]
            #self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.Ndisp_est]
            return

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]



        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        
        with tf.variable_scope('model'):
                self.D_left_real = self.discriminator(self.left_pyramid[0], self.options, reuse=False, name="discriminatorB")
                self.D_left_fake = self.discriminator(self.left_est[0], self.options, reuse=True, name="discriminatorB")

                self.D_right_real = self.discriminator(self.right_pyramid[0], self.options, reuse=False, name="discriminatorA")
                self.D_right_fake = self.discriminator(self.right_est[0], self.options, reuse=True, name="discriminatorA")
    
        t_vars = tf.trainable_variables()
        self.discrA_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.discrB_vars = [var for var in t_vars if 'discriminatorB' in var.name]

        if self.mode == 'test':
                #self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_left_est]
                return

    def disp_output(self):
        disp_out = tf.reduce_mean(tf.concat([self.disp_left_est[0], self.disp_right_est[0]], 3), 3, keep_dims=True)
        return disp_out

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1 (identity)
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)


            # L1 (cycle)
            self.l1_cycle_b = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_cycle_backward = [tf.reduce_mean(l) for l in self.l1_cycle_b]
            if self.branch == 'a2b':
                self.cycle_loss = tf.add_n(self.l1_cycle_backward)
            elif self.branch == 'b2a':
                self.cycle_loss = 0
            else:
                self.cycle_loss = tf.add_n(self.l1_cycle_backward)

            #SUM
            self.image_loss_right = [self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.l1_reconstruction_loss_left[i]  for i in range(4)]
            #self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)
            if self.branch == 'b2a':
                self.image_loss = tf.add_n(self.image_loss_left)
            elif self.branch == 'a2b':
                self.image_loss = tf.add_n(self.image_loss_right)
            else:
                self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISCRIMINATOR
            # left
            self.d_loss_left_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_left_real, tf.ones_like(self.D_left_real)))
            self.d_loss_left_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_left_fake, tf.ones_like(self.D_left_fake)))
            self.g_loss_left_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_left_fake, tf.ones_like(self.D_left_fake)))
            self.d_loss_left = (self.d_loss_left_real + self.d_loss_left_fake) 
            # right
            self.d_loss_right_real = self.criterionGAN(self.D_right_real, tf.ones_like(self.D_right_real))
            self.d_loss_right_fake = self.criterionGAN(self.D_right_fake, tf.zeros_like(self.D_right_fake))
            self.d_loss_right = (self.d_loss_right_real + self.d_loss_right_fake) / 2
  


            # # TOTAL LOSS
            self.total_loss = self.image_loss + 0.1*self.g_loss_left_fake
            self.discr_loss = 0.0001 * self.d_loss_left

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):

            for i in range(4):
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i] , max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.histogram('disp_left_est_' + str(i), self.disp_left_est[i], collections=self.model_collection)
                tf.summary.histogram('disp_right_est_' + str(i), self.disp_right_est[i], collections=self.model_collection)
                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)

            tf.summary.scalar('d_loss', self.d_loss_left, collections=self.model_collection)
            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)
