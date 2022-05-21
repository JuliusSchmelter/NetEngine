import csv
from glob import glob
from random import randrange


# get all patches
glob_res = glob(".\\data\\data\\*")
all = [p.split("\\")[-1] for p in glob_res]

# get snow covered patches
with open(".\\data\\patches_with_seasonal_snow.csv") as file:
    snow = set([p[0] for p in list(csv.reader(file))])

# get cloudy patches
with open(".\\data\\patches_with_cloud_and_shadow.csv") as file:
    clouds = set([p[0] for p in list(csv.reader(file))])

# get clear patches
clear = [p for p in all if p not in snow and p not in clouds]

# split into training, test and validation data
training = []
test = []
validation = []

for patch in clear:
    random = randrange(0, 100)
    if random < 60:
        training.append(patch)
    elif random < 80:
        test.append(patch)
    else:
        validation.append(patch)

print(f"{len(all) = }")
print(f"{len(clear) = }")
print(f"{len(training) = }")
print(f"{100*len(training)/len(clear) = }")
print(f"{len(test) = }")
print(f"{100*len(test)/len(clear) = }")
print(f"{len(validation) = }")
print(f"{100*len(validation)/len(clear) = }")

# save as csv
with open(".\\data\\training.csv", "w") as file:
    csv.writer(file, delimiter="\n").writerow(training)

with open(".\\data\\test.csv", "w") as file:
    csv.writer(file, delimiter="\n").writerow(test)

with open(".\\data\\validation.csv", "w") as file:
    csv.writer(file, delimiter="\n").writerow(validation)
