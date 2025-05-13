import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class DilatedConvBlock(tf.keras.layers.Layer):
    def __init__(self, dilation_rate, filters=64, kernel_size=3):
        super().__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding='same',
                                  dilation_rate=dilation_rate, activation='relu')

    def call(self, x):
        return self.conv(x)

'''
Input: 
    grayscale image + previous hidden state (combined as 65 channels)
Output:
    pout: pixel-wise action logits (policy)
    vout: pixel-wise value estimate
    h_t: updated hidden state for next time step (ConvGRU-style)
'''
class MyFcn(Model):
    def __init__(self, n_actions):
        super().__init__()
        he_init = tf.keras.initializers.HeNormal()

        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')  # 1-channel assumed preprocessed

        # Shared Feature Extractor
        # Dilated Conv - to capture wider spatial context without increasing kernel size
        self.diconv2 = DilatedConvBlock(dilation_rate=2)
        self.diconv3 = DilatedConvBlock(dilation_rate=3)
        self.diconv4 = DilatedConvBlock(dilation_rate=4)

        # Policy Branch
        self.diconv5_pi = DilatedConvBlock(dilation_rate=3)
        self.diconv6_pi = DilatedConvBlock(dilation_rate=2)

        # ConvGRU-like gates
        self.conv7_Wz = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=he_init)
        self.conv7_Uz = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=he_init)
        self.conv7_Wr = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=he_init)
        self.conv7_Ur = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=he_init)
        self.conv7_W = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=he_init)
        self.conv7_U = layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer=he_init)

        self.conv8_pi = layers.Conv2D(n_actions, 3, padding='same')  # logits

        # Value Branch
        self.diconv5_V = DilatedConvBlock(dilation_rate=3)
        self.diconv6_V = DilatedConvBlock(dilation_rate=2)
        self.conv7_V = layers.Conv2D(1, 3, padding='same')  # scalar value output

    def call(self, x):
        """x shape: (batch_size, height, width, channels) with channels=65 (1 input + 64 GRU hidden)"""
        x_input = x[:, :, :, :1]     # grayscale input
        h_t1 = x[:, :, :, 1:]        # previous hidden state (64 channels)

        h = self.conv1(x_input)
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)

        # --- Policy Branch ---
        h_pi = self.diconv5_pi(h)
        x_t = self.diconv6_pi(h_pi)

        z_t = tf.sigmoid(self.conv7_Wz(x_t) + self.conv7_Uz(h_t1))      # Update gate
        r_t = tf.sigmoid(self.conv7_Wr(x_t) + self.conv7_Ur(h_t1))      # Forget gate
        h_tilde = tf.tanh(self.conv7_W(x_t) + self.conv7_U(r_t * h_t1)) # Candidate activation
        h_t = (1 - z_t) * h_t1 + z_t * h_tilde                          # new hidden state

        pout = self.conv8_pi(h_t)  # logits

        # --- Value Branch ---
        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)

        return pout, vout, h_t
