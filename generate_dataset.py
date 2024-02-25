import glob
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt


class Mydataset(data.Dataset):
    def __init__(self, root):
        self.imgs_path = root

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        return img_path

    def __len__(self):
        return len(self.imgs_path)


all_imgs_path = glob.glob(r'F:\animal\*\*.jpg')
for var in all_imgs_path:
    print(var)

weather_dataset = Mydataset(all_imgs_path)
print(len(weather_dataset))
print(weather_dataset[12:14])
wheather_datalodaer = torch.utils.data.DataLoader(weather_dataset, batch_size=3)
print(next(iter(wheather_datalodaer)))

species = ['cat', 'bird', 'fish']
species_to_id = dict((c, i) for i, c in enumerate(species))
print(species_to_id)
id_to_species = dict((v, k) for k, v in species_to_id.items())
print(id_to_species)
all_labels = []
for img in all_imgs_path:
    for i, c in enumerate(species):
        if c in img:
            all_labels.append(i)
print(all_labels)

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])


class Mydatasetpro(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)
        data = self.transforms(pil_img)
        return data, label

    def __len__(self):
        return len(self.imgs)


BATCH_SIZE = 10
weather_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
wheather_datalodaer = data.DataLoader(
    weather_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

imgs_batch, labels_batch = next(iter(wheather_datalodaer))
print(imgs_batch.shape)

plt.figure(figsize=(12, 8))
for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    img = img.permute(1, 2, 0).numpy()
    plt.subplot(2, 3, i + 1)
    plt.title(id_to_species.get(label.item()))
    plt.imshow(img)
plt.show()

index = np.random.permutation(len(all_imgs_path))

all_imgs_path = np.array(all_imgs_path)[index]
all_labels = np.array(all_labels)[index]

s = int(len(all_imgs_path) * 0.8)
print(s)

train_imgs = all_imgs_path[:s]
train_labels = all_labels[:s]
test_imgs = all_imgs_path[s:]
test_labels = all_labels[s:]

train_ds = Mydatasetpro(train_imgs, train_labels, transform)  # TrainSet TensorData
test_ds = Mydatasetpro(test_imgs, test_labels, transform)  # TestSet TensorData
train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TrainSet Labels
test_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)  # TestSet Labels
