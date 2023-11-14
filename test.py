import netengine
import math
from linetimer import CodeTimer

N_TEST = 10000
N_TRAIN = 60000

LAYOUT = [28 * 28, 500, 200, 60, 10]
ETA = 0.1
ETA_BIAS = 0.02

MINI_BATCHES = 16
MINI_BATCH_SIZE = 1024


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
print(net)

# Train and test.
trained = 0
max_acc = 0
max_acc_at = 0
while True:
    net.train(train_images, train_labels, MINI_BATCHES, MINI_BATCH_SIZE, trained % N_TRAIN)
    trained += MINI_BATCHES * MINI_BATCH_SIZE

    acc = 100 * net.test(test_images, test_labels)
    if acc > max_acc:
        max_acc = acc
        max_acc_at = trained

    print(f"trained: {trained} | accuracy: {acc:.2f}%")
    print(f"maximum accuracy: {max_acc:.2f}% | at trained: {max_acc_at}")


# Results:
#
# layout: 784 | 500 | 200 | 60 | 10
# parameters: 505370
# eta: 0.1
# eta_bias: 0.02
# maximum accuracy: 96.97% | at trained: 21,364,736
