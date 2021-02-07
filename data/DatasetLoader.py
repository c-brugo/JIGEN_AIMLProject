import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random, randint


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)

def image_tensor_transformer():
    img_tr = [transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)


class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset, img_transformer=None, jig_transformer=None, beta=0, odd_one_out=False):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self._jigsaw_transformer = jig_transformer
        self._tensor_transformer = image_tensor_transformer()
        self.beta = beta
        self.odd_one_out = odd_one_out

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)
        permutation = 0

        if random() < self.beta:
            if self.odd_one_out:
                index_image_odd = randint(0, len(self.names)-1)
                img2 = Image.open(self.data_path + '/' + self.names[index_image_odd]).convert('RGB')
                img2 = self._image_transformer(img)
                img, permutation = self._jigsaw_transformer(img, img2)
            else:
                img, permutation = self._jigsaw_transformer(img)

        img = self._tensor_transformer(img)

        return img, int(self.labels[index]), permutation

    def __len__(self):
        return len(self.names)



class TestDataset(Dataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index])


