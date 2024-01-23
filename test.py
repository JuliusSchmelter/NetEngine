import netengine
from linetimer import CodeTimer

N_TEST = 10000
N_TRAIN = 60000

LAYOUT = [28 * 28, 100, 10]
LAYOUT = [28 * 28, 500, 200, 60, 10]
ETA = 0.1
ETA_BIAS = 0.02

BATCH_SIZE = 60000

# Define target vectors.
targets = []
for i in range(10):
    target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target[i] = 1
    targets.append(target)

# Load test data.
with CodeTimer("load test data"):
    with open("data/mnist/test_labels", "rb") as file:
        # Skip header.
        file.seek(8)
        test_labels = [targets[int(i)] for i in file.read()][:N_TEST]

    test_images = []
    with open("data/mnist/test_images", "rb") as file:
        # Skip header.
        file.seek(16)
        for i in range(N_TEST):
            # Scale to [0, 1].
            test_images.append([pixel / 255.0 for pixel in file.read(28 * 28)])

# Load training data.
with CodeTimer("load training data"):
    with open("data/mnist/train_labels", "rb") as file:
        # Skip header.
        file.seek(8)
        train_labels = [targets[int(i)] for i in file.read()][:N_TRAIN]

    train_images = []
    with open("data/mnist/train_images", "rb") as file:
        # Skip header.
        file.seek(16)
        for i in range(N_TRAIN):
            # Scale to [0, 1].
            train_images.append([pixel / 255.0 for pixel in file.read(28 * 28)])


# Get Net instance.
with CodeTimer("create net"):
    net = netengine.Net(LAYOUT, ETA, ETA_BIAS, True)

print()
print(net)
print(f"\n{net.cuda_enabled() = }\n")

# Train and test.
trained = 0
acc = 0
max_acc = 0
max_acc_at = 0
while True:
    net.set_eta(ETA * (1 - acc))
    net.set_eta_bias(ETA_BIAS * (1 - acc))

    with CodeTimer("train"):
        net.train(train_images, train_labels, BATCH_SIZE, trained % N_TRAIN)

    trained += BATCH_SIZE

    with CodeTimer("test"):
        acc = net.test(test_images, test_labels)

    if acc > max_acc:
        max_acc = acc
        max_acc_at = trained

    print(f"accuracy: {100 * acc:.2f}% | trained: {trained}")
    print(f"maximum accuracy: {100 * max_acc:.2f}% | at trained: {max_acc_at}\n")


# Results:
#
# layout: 784 | 500 | 200 | 60 | 10
# parameters: 505370
# eta: 0.1
# eta_bias: 0.02
# maximum accuracy: 96.97% | at trained: 21,364,736
