import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F
from torch.utils.data import Dataset


def random_rot_flip(image, image1, image2, label):
    k, axis = np.random.randint(0, 4), np.random.randint(0, 2)
    return tuple(np.flip(np.rot90(x, k), axis=axis).copy() for x in (image, image1, image2, label))


def random_rotate(image, image1, image2, label):
    angle = np.random.uniform(-20, 20)
    return tuple(
        ndimage.rotate(x, angle, order=0 if i == 3 else 1, reshape=False, mode="constant", cval=0, prefilter=False)
        for i, x in enumerate((image, image1, image2, label))
    )


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.02):
    images = tuple(np.clip(x + np.random.normal(mean, std, x.shape), 0, 1) for x in (image, image1, image2))
    return (*images, label)



class ResizeGenerator:
    """Deterministic, alignment-preserving image resize and tensor conversion."""

    def __init__(self, output_size):
        if len(output_size) != 2 or any(int(v) <= 0 for v in output_size):
            raise ValueError("output_size must contain two positive dimensions.")
        self.output_size = tuple(int(v) for v in output_size)

    def __call__(self, sample):
        result = dict(sample)
        shape = sample["label"].shape
        if len(shape) != 2:
            raise ValueError("Slice transforms require 2D samples; volume resize belongs in inference.")
        for key in ("image", "image1", "image2", "label"):
            value = np.asarray(sample[key])
            if value.shape != shape:
                raise ValueError("Modalities and mask must have identical spatial shapes.")
            dtype = np.int64 if key == "label" else np.float32
            tensor = torch.from_numpy(np.array(value, dtype=dtype, copy=True, order="C"))
            if shape != self.output_size:
                if key == "label":
                    tensor = F.interpolate(tensor[None, None].float(), size=self.output_size, mode="nearest")[0, 0].long()
                else:
                    tensor = F.interpolate(tensor[None, None], size=self.output_size, mode="bilinear", align_corners=False)[0, 0]
            result[key] = tensor if key == "label" else tensor.unsqueeze(0)
        return result


class RandomGenerator(ResizeGenerator):
    """Shared spatial warps and independent intensity variation on [0,1] images.

    Translation pads with background and never wraps structures at image edges.
    """

    def __call__(self, sample):
        result = dict(sample)
        images = [np.asarray(sample[k]).copy() for k in ("image", "image1", "image2")]
        label = np.asarray(sample["label"]).copy()
        if random.random() < 0.5:
            *images, label = random_rot_flip(*images, label)
        elif random.random() < 0.5:
            *images, label = random_rotate(*images, label)
        if random.random() < 0.5:
            shift = (random.randint(-5, 5), random.randint(-5, 5))
            images = [ndimage.shift(x, shift, order=1, mode="constant", cval=0, prefilter=False) for x in images]
            label = ndimage.shift(label, shift, order=0, mode="constant", cval=0, prefilter=False)
        for i, image in enumerate(images):
            if random.random() < 0.5:
                image = np.clip(image * random.uniform(0.8, 1.2), 0, 1)
            if random.random() < 0.5:
                image = np.clip(image, 0, 1) ** random.uniform(0.8, 1.2)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0, 0.02, image.shape), 0, 1)
            result[("image", "image1", "image2")[i]] = image
        result["label"] = label
        return super().__call__(result)


class Myops_dataset(Dataset):
    def __init__(self, base_dir, base_dir1, base_dir2, list_dir, split, transform=None):
        """
            base_dir: bSSFP modal
            base_dir1: LGE modal
            base_dir2: T2w modal
        """
        self.transform = transform  
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.data_dir1 = base_dir1
        self.data_dir2 = base_dir2

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data_path_1 = os.path.join(self.data_dir1, slice_name+'.npz')
            data_path_2 = os.path.join(self.data_dir2, slice_name+'.npz')
            data = np.load(data_path)
            data_1 = np.load(data_path_1)
            data_2 = np.load(data_path_2)
            image, label = data['image'], data['label']
            image1 = data_1['image']
            image2 = data_2['image']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            filepath_1 = self.data_dir1 + "/{}.npy.h5".format(vol_name)
            filepath_2 = self.data_dir2 + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            data1 = h5py.File(filepath_1)
            data2 = h5py.File(filepath_2)
            image, label = data['image'][:], data['label'][:]
            image1 = data1['image'][:]
            image2 = data2['image'][:]

        sample = {'image': image, 'image1':image1, "image2":image2, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
