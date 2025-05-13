import numpy as np
import cv2


'''
    Maintains current image state
    Applies pixel-wise actions (e.g. filters or value tweaks)
    Updates the input tensor fed into the model (image + GRU state)
'''
class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.move_range = move_range

    def reset(self, x, n):
        self.image = x + n
        size = self.image.shape
        prev_state = np.zeros((size[0], 64, size[2], size[3]), dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:, :self.image.shape[1], :, :] = self.image

    # Take actions and update states
    def step(self, act, inner_state):
        if act.ndim == 2:
            act = np.expand_dims(act, axis=0)
        if act.ndim == 3:
            act = np.expand_dims(act, axis=1)  # (B, 1, H, W)

        neutral = (self.move_range - 1) / 2
        move = (act.astype(np.float32) - neutral) / 255.0

        moved_image = self.image + move

        # Prepare filtered outputs
        b, c, h, w = self.image.shape
        filtered_outputs = {
            self.move_range: np.zeros_like(self.image),         # Gaussian
            self.move_range + 1: np.zeros_like(self.image),     # Bilateral
            self.move_range + 2: np.zeros_like(self.image),     # Median
            self.move_range + 3: np.zeros_like(self.image),     # Gaussian (stronger)
            self.move_range + 4: np.zeros_like(self.image),     # Bilateral (stronger)
            self.move_range + 5: np.zeros_like(self.image),     # Box
        }

        for i in range(b):
            img = self.image[i, 0]
            if np.any(act[i, 0] == self.move_range):
                filtered_outputs[self.move_range][i, 0] = cv2.GaussianBlur(img, (5, 5), 0.5)
            if np.any(act[i, 0] == self.move_range + 1):
                filtered_outputs[self.move_range + 1][i, 0] = cv2.bilateralFilter(img, 5, 10, 5)
            if np.any(act[i, 0] == self.move_range + 2):
                filtered_outputs[self.move_range + 2][i, 0] = cv2.medianBlur(img, 5)
            if np.any(act[i, 0] == self.move_range + 3):
                filtered_outputs[self.move_range + 3][i, 0] = cv2.GaussianBlur(img, (5, 5), 1.5)
            if np.any(act[i, 0] == self.move_range + 4):
                filtered_outputs[self.move_range + 4][i, 0] = cv2.bilateralFilter(img, 5, 50, 5)
            if np.any(act[i, 0] == self.move_range + 5):
                filtered_outputs[self.move_range + 5][i, 0] = cv2.boxFilter(img, -1, (5, 5))

        # Apply filters selectively
        self.image = moved_image
        for key, filtered in filtered_outputs.items():
            self.image = np.where(act == key, filtered, self.image)

        # Update tensor
        self.tensor[:, :self.image.shape[1], :, :] = self.image
        inner_state = np.transpose(inner_state, (2, 0, 1))  # (64, H, W)
        inner_state = np.expand_dims(inner_state, axis=0)   # (1, 64, H, W)
        self.tensor[:, -64:, :, :] = inner_state

