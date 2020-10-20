import os
import zipfile

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

IMG_FILE_TYPES = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


class PatchMiceDatasetFromTensor(Dataset):
    def __init__(self, data_root, max_dataset_size=float('inf'), is_train=True,
                 tumor_threshold=0.2, is_3d=False):
        Dataset.__init__(self)
        self.tumor_threshold = tumor_threshold
        if is_train:
            self.dir = os.path.join(data_root, 'train')
        else:
            self.dir = os.path.join(data_root, 'test')
        self.is_3d = is_3d
        self.dataset = self.make_dataset_from_tensor(self.dir, max_dataset_size)
        self.size = len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index % self.size]

    def __len__(self):
        return self.size

    def get_obj_data(self, data_path):
        data = torch.load(data_path)
        label = 1 if data['tumor_percentage'] > self.tumor_threshold else 0
        image = data['image']
        if self.is_3d:
            image = image.permute(2, 0, 1).unsqueeze(0)

        return {'image': image,
                'mask': data['mask'] / 255,
                'tumor_percentage': data['tumor_percentage'],
                'slice_range': data['slice_range'],
                'patch': data['patch'],
                'mouse_name': data['mouse_name'],
                'label': label,
                'path': data_path}

    def make_dataset_from_tensor(self, dir, max_dataset_size=float('inf')):
        true_label_data, false_label_data = [], []
        for mice_name in os.listdir(dir):
            if len(true_label_data) >= max_dataset_size and len(false_label_data) >= max_dataset_size:
                break
            for root, _, file_names in os.walk(os.path.join(dir, mice_name)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    file_data = self.get_obj_data(path)
                    label = file_data['label']
                    if label == 1:
                        true_label_data.append(file_data)
                    else:
                        false_label_data.append(file_data)

        data = false_label_data[:min(max_dataset_size, len(false_label_data))] + \
               true_label_data[:min(max_dataset_size, len(true_label_data))]
        return data


class PatchMiceDataset(Dataset):
    def __init__(self, data_root, max_dataset_size=float('inf'), is_train=True, depth=None):
        Dataset.__init__(self)
        if is_train:
            self.dir = os.path.join(data_root, 'train')
        else:
            self.dir = os.path.join(data_root, 'test')
        dataset_no_tumor = sorted(make_dataset(self.dir, 'no tumor', max_dataset_size, depth))
        dataset_tumor = sorted(make_dataset(self.dir,  'tumor', max_dataset_size, depth))

        self.labels = [0] * len(dataset_no_tumor) + [1] * len(dataset_tumor)

        self.dataset = dataset_no_tumor + dataset_tumor
        self.size = len(self.dataset)
        self.transform = get_transform()

    def __getitem__(self, index):
        image_path = self.dataset[index % self.size]
        _, patch, _, slice_num, _ = os.path.basename(image_path).replace('.', ' ').split(' ')
        mice_name = image_path.split(os.sep)[-3]
        label = self.labels[index % self.size]

        image = Image.open(image_path)
        image = self.transform(image)

        return {'image': image, 'label': label, 'slice': slice_num, 'patch': patch, 'mice': mice_name}

    def __len__(self):
        return self.size


def extract_slice_patch_from_filename(file_name):
    file_name_parts = file_name.replace('.', ' ').split(' ')
    slice_num = int(file_name_parts[-2])
    patch = file_name_parts[-4]
    return slice_num, patch


def make_dataset(dir, folder_name, max_dataset_size=float('inf'), depth=None):
    images = []
    image_slices = []  # only for depth option
    for mice_name in os.listdir(dir):
        for root, _, file_names in os.walk(os.path.join(dir, mice_name, folder_name)):
            for file_name in file_names:
                slice_num, patch = extract_slice_patch_from_filename(file_name)
                if is_image(file_name):
                    path = os.path.join(root, file_name)
                    if depth:
                        if not image_slices:
                            image_slices.append(path)
                        else:
                            _, prev_patch, _, prev_slice_num, _ = os.path.basename(image_slices[-1])\
                                                                    .replace('.', ' ').split(' ')
                            prev_slice_num = int(prev_slice_num)
                            if len(image_slices) < depth and prev_slice_num + 1 == slice_num and prev_patch == patch:
                                image_slices.append(path)
                            elif len(image_slices) == depth:
                                images.append(image_slices)
                                image_slices = [path]
                            elif len(image_slices) < depth and (prev_slice_num + 1 != slice_num or prev_patch != patch):
                                image_slices = [path]
                    else:
                        images.append(path)
    return images[:min(max_dataset_size, len(images))]


def is_image(file_name):
    return any(file_name.endswith(extension) for extension in IMG_FILE_TYPES)


def get_transform():
    values = [transforms.Grayscale(1),
              transforms.ToTensor(),
              transforms.Normalize((0.5,), (0.5,))]
    return transforms.Compose(values)