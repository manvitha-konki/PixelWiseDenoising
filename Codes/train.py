import os
import numpy as np
import cv2
import time
import tensorflow as tf

from MyFCN import MyFcn
from pixelwise_a3c import PixelWiseA3CAgent
from mini_batch_loader import MiniBatchLoader
import State
import sys

'''
training and testing of the A3C agent.
Loads data, simulates environment, stores transitions, computes rewards.
Periodically logs, evaluates, and saves model weights
'''

# Paths and hyperparameters
TRAINING_DATA_PATH = r"C:\Users\91898\Desktop\gitfilesFinal\gitfilesFinal\gitfiles\pixelRL\training_BSD68.txt"
TESTING_DATA_PATH = r"C:\Users\91898\Desktop\gitfilesFinal\gitfilesFinal\gitfiles\pixelRL\testing.txt"
IMAGE_DIR_PATH = r"C:\Users\91898\Desktop\gitfilesFinal\gitfilesFinal\gitfiles\pixelRL\BSD68\gray"
SAVE_PATH = "./model_tf/denoise_myfcn_"
RESULT_PATH = "./resultimage_tf/"

LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 1
N_EPISODES = 100000
EPISODE_LEN = 5
SNAPSHOT_EPISODES = 3000
TEST_EPISODES = 3000
GAMMA = 0.95

MEAN = 0
SIGMA = 15
N_ACTIONS = 9
MOVE_RANGE = 7
CROP_SIZE = 70

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

def test(loader, agent, fout):
    import os
    import cv2
    import numpy as np

    sum_psnr = 0
    sum_input_psnr = 0
    sum_reward = 0
    test_data_size = len(loader.testing_path_infos)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)

    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        upper = min(i + TEST_BATCH_SIZE, test_data_size)
        indices = np.arange(i, upper)
        raw_x = loader.load_testing_data(indices)

        raw_n = np.random.normal(MEAN, SIGMA, raw_x.shape).astype(np.float32) / 255
        current_state.reset(raw_x, raw_n)
        reward = np.zeros_like(raw_x) * 255

        for t in range(EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        # --- Prepare outputs ---
        I = np.clip(raw_x[0], 0, 1)                  # Ground truth
        N = np.clip(raw_x[0] + raw_n[0], 0, 1)       # Noisy input
        P = np.clip(current_state.image[0], 0, 1)    # Predicted output

        I_uint8 = (I * 255 + 0.5).astype(np.uint8)
        N_uint8 = (N * 255 + 0.5).astype(np.uint8)
        P_uint8 = (P * 255 + 0.5).astype(np.uint8)

        # Convert (1, H, W) → (H, W)
        if I_uint8.ndim == 3 and I_uint8.shape[0] == 1:
            I_uint8 = I_uint8[0]
            N_uint8 = N_uint8[0]
            P_uint8 = P_uint8[0]

        # Calculate PSNR
        input_psnr = cv2.PSNR(N_uint8, I_uint8)
        output_psnr = cv2.PSNR(P_uint8, I_uint8)

        sum_input_psnr += input_psnr
        sum_psnr += output_psnr

        # Save images
        cv2.imwrite(os.path.join(RESULT_PATH, f'{i}_clean.png'), I_uint8)
        cv2.imwrite(os.path.join(RESULT_PATH, f'{i}_input.png'), N_uint8)
        cv2.imwrite(os.path.join(RESULT_PATH, f'{i}_output.png'), P_uint8)

        # Log per-image PSNR
        print(f"[{i}] Input PSNR: {input_psnr:.2f}, Output PSNR: {output_psnr:.2f}")
        fout.write(f"[{i}] Input PSNR: {input_psnr:.2f}, Output PSNR: {output_psnr:.2f}\n")

    avg_reward = sum_reward * 255 / test_data_size
    avg_input_psnr = sum_input_psnr / test_data_size
    avg_output_psnr = sum_psnr / test_data_size

    print(f"\nTest total reward: {avg_reward:.2f}")
    print(f"Avg Input PSNR: {avg_input_psnr:.2f}")
    print(f"Avg Output PSNR: {avg_output_psnr:.2f}")
    fout.write(f"\nTest total reward: {avg_reward:.2f}\n")
    fout.write(f"Avg Input PSNR: {avg_input_psnr:.2f}, Avg Output PSNR: {avg_output_psnr:.2f}\n")
    fout.flush()

def main(fout):
    loader = MiniBatchLoader(TRAINING_DATA_PATH, TESTING_DATA_PATH, IMAGE_DIR_PATH, CROP_SIZE)
    model = MyFcn(n_actions=N_ACTIONS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    agent = PixelWiseA3CAgent(model, optimizer, gamma=GAMMA, t_max=EPISODE_LEN)
    current_state = State.State((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)

    train_data_size = len(loader.training_path_infos)
    indices = np.random.permutation(train_data_size)
    i = 0

    for episode in range(1, N_EPISODES + 1):
        print(f"Episode {episode}")
        fout.write(f"Episode {episode}\n")
        sys.stdout.flush()

        # r = indices[i:i+TRAIN_BATCH_SIZE]
        r = indices[i:min(i+TRAIN_BATCH_SIZE, train_data_size)]

        # Skip if we can't make a full batch
        if len(r) < TRAIN_BATCH_SIZE:
            i = 0
            indices = np.random.permutation(train_data_size)
            continue

        raw_x = loader.load_training_data(r)
        raw_n = np.random.normal(MEAN, SIGMA, raw_x.shape).astype(np.float32) / 255
        current_state.reset(raw_x, raw_n)
        reward = np.zeros_like(raw_x)
        sum_reward = 0

        for t in range(EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            # reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            # PSNR-based reward
            def psnr(img1, img2):
                mse = np.mean((img1 - img2) ** 2)
                if mse == 0:
                    return 100.0
                PIXEL_MAX = 1.0
                return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

            psnr_prev = psnr(raw_x[0, 0], previous_image[0, 0])
            psnr_curr = psnr(raw_x[0, 0], current_state.image[0, 0])
            reward_value = psnr_curr - psnr_prev
            reward = np.ones_like(raw_x) * reward_value

            agent.store_transition(
                current_state.tensor,
                action,
                reward_value,
                model(tf.convert_to_tensor(current_state.tensor[None, ...], dtype=tf.float32))[1],
                model(current_state.tensor[None, ...])[0]
            )
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        agent.update(current_state.tensor)
        print(f"Train total reward: {sum_reward * 255}")
        fout.write(f"Train total reward: {sum_reward * 255}\n")
        fout.flush()

        if episode % TEST_EPISODES == 0:
            test(loader, agent, fout)

        if episode % SNAPSHOT_EPISODES == 0:
            model.save_weights(SAVE_PATH + f"{episode}")

        i = (i + TRAIN_BATCH_SIZE) % train_data_size
        if i + 2 * TRAIN_BATCH_SIZE >= train_data_size:
            indices = np.random.permutation(train_data_size)
            i = train_data_size - TRAIN_BATCH_SIZE

        new_lr = LEARNING_RATE * ((1 - episode / N_EPISODES) ** 0.5)
        optimizer.learning_rate.assign(new_lr)

    print("Training complete — running final test...")
    test(loader, agent, fout)


fout = open('log_tf.txt', "w")
start = time.time()
main(fout)
end = time.time()
print(f"{end - start:.2f}[s], {(end - start)/60:.2f}[m], {(end - start)/3600:.2f}[h]")
fout.write(f"{end - start:.2f}[s], {(end - start)/60:.2f}[m], {(end - start)/3600:.2f}[h]\n")
fout.close()
