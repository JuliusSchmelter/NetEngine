import netengine
import math
from linetimer import CodeTimer

N_TEST = 10000
N_TRAIN = 60000

LAYOUT = [28 * 28, 500, 200, 100, 10]
ETA = 0.1
ETA_BIAS = 0.02

MINI_BATCHES = 32
MINI_BATCH_SIZE = 2048


# Define target vectors.
targets = []
for i in range(10):
    target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target[i] = 1
    targets.append(target)

# Load test data.
with CodeTimer("load test data"):
    with open("data/mnist/test_labels", "rb") as file:
        file.seek(8)  # skip header
        test_labels = [targets[int(i)] for i in file.read()][:N_TEST]

    test_images = []
    with open("data/mnist/test_images", "rb") as file:
        file.seek(16)  # skip header
        for i in range(N_TEST):
            test_images.append([pixel / 255.0 for pixel in file.read(28 * 28)])  # scale to [0, 1]

# Load training data.
with CodeTimer("load training data"):
    with open("data/mnist/train_labels", "rb") as file:
        file.seek(8)  # skip header
        train_labels = [targets[int(i)] for i in file.read()][:N_TRAIN]

    train_images = []
    with open("data/mnist/train_images", "rb") as file:
        file.seek(16)  # skip header
        for i in range(N_TRAIN):
            train_images.append([pixel / 255.0 for pixel in file.read(28 * 28)])  # scale to [0, 1]


# Get Net instance.
net = netengine.Net(LAYOUT, ETA, ETA_BIAS)
print(f"number of parameters: {net.n_parameters()}")

# Train and test.
trained = 0
while True:
    net.train(train_images, train_labels, MINI_BATCHES, MINI_BATCH_SIZE, trained % N_TRAIN)
    trained += MINI_BATCHES * MINI_BATCH_SIZE
    print(f"trained: {trained} | accuracy: {100 * net.test(test_images, test_labels):.2f}%")
