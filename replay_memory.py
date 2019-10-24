import random
import numpy as np
import os


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in ind:
            s, a, r, s2, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            next_state.append(np.array(s2, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            done.append(np.array(d, copy=False))

        return (np.array(state),
                np.array(action),
                np.array(reward),
                np.array(next_state),
                np.array(done))

    # def sample(self, batch_size):
    #     batch = random.sample(self.buffer, batch_size)
    #     state, action, reward, next_state, done = map(np.stack, zip(*batch))
    #     return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        if not os.path.exists("./buffers"):
            os.makedirs("./buffers")
        np.save("./buffers/" + filename + ".npy", self.buffer)

    def load(self, filename):
        self.buffer = np.load("./buffers/" + filename + ".npy", allow_pickle=True)
        print("load buffer: " + filename + ".npy")

