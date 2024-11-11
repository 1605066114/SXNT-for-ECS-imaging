###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG','.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images_sxt = []
    images_em = []
    root_sxt = dir+'/sxt'
    root_em = dir+'/em'
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    '''for root, _, fnames in sorted(os.listdir(dir+'/sxt')):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images_sxt.append(path)'''
    sxt_files = os.listdir(dir+'/sxt')

    for fname in sxt_files:
    
        images_sxt.append(os.path.join(root_sxt, fname))
        images_em.append(os.path.join(root_em, fname.replace('sxt', 'em')))

    return images_sxt, images_em


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
