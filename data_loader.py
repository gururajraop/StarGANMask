from torch.utils import data
from torchvision.transforms import functional as TF
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class DataLoaderClass(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transforms, mode, crop_size, image_size):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transforms = transforms
        self.mode = mode
        self.crop_size = crop_size
        self.image_size = image_size
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def get_transforms(self, image, target):
      if self.mode == 'train':
        if random.random() > 0.5:
          image = TF.hflip(image)
          target = TF.hflip(target)

      crop = T.CenterCrop(self.crop_size)
      image = crop(image)
      target = crop(target)

      resize = T.Resize(self.image_size)
      image = resize(image)
      target = resize(target)

      image = TF.to_tensor(image)
      target = TF.to_tensor(target)

      normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
      image = normalize(image)
      target = normalize(target)

      return image, target


    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        folders = os.listdir(self.image_dir)
        for i, folder in enumerate(folders):
          if folder == 'train':
            print('Processing training dataset')
            classes = os.listdir(os.path.join(self.image_dir, folder))
            for cls in classes:
              label = [1] if cls == 'corrected' else [0]
              files = os.listdir(os.path.join(self.image_dir, folder, cls))
              random.seed(1234)
              random.shuffle(files)
              for filename in files:
                self.train_dataset.append([filename, cls, label])
          elif folder == 'test':
            print('Processing testing dataset')
            classes = os.listdir(os.path.join(self.image_dir, folder))
            for cls in classes:
              label = [1] if cls == 'corrected' else [0]
              files = os.listdir(os.path.join(self.image_dir, folder, cls))
              random.seed(1234)
              random.shuffle(files)
              for filename in files:
                self.test_dataset.append([filename, cls, label])
            
        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, cls, label = dataset[index]
        target_cls = 'original' if cls == 'corrected' else 'corrected'
        image = Image.open(os.path.join(self.image_dir, self.mode, cls, filename))
        filename = filename.replace('d65', 'img') if cls == 'corrected' else filename.replace('img', 'd65')
        target = Image.open(os.path.join(self.image_dir, self.mode, target_cls, filename))
        image, target = self.get_transforms(image, target)
        if self.mode == 'train'
          return image, target, torch.LongTensor(label)
        else:
          return image, target, torch.LongTensor(label), filename

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        #dataset = ImageFolder(image_dir, transform)
        dataset = DataLoaderClass(image_dir, transform, mode, crop_size, image_size)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
