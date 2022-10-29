import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import json
import numpy as np
import os
from collections import defaultdict

# This file is a modified version of the LEAF code provided here https://github.com/TalwalkarLab/leaf
def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


class CelebA(Dataset):
    """ CelebA dataset."""

    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.root_dir = root_dir
        self.dataset_name = 'celeba'
        train_data_dir = os.path.join(self.root_dir, self.dataset_name, 'data', 'train')
        test_data_dir = os.path.join(self.root_dir, self.dataset_name, 'data', 'test')
        _, _, train_data, test_data = read_data(train_data_dir, test_data_dir)
        self.img_dir = os.path.join(self.root_dir, self.dataset_name, 'data', 'raw', 'img_align_celeba')

        if train:
            self.data_dir = train_data_dir
            self.data = train_data
        else:
            self.data_dir = test_data_dir
            self.data = test_data
        self.data_keys = list(self.data.keys())
        self.data_keys.sort()

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        idx = self.data_keys[idx]
        img_names = self.data[idx]['x']
        imgs = []
        for img_name in img_names:
            img_name = os.path.join(self.img_dir, img_name)
            img = io.imread(img_name)/256.0
            if self.transform:
                img = self.transform(img)
            imgs.append( img )

        imgs = np.array( imgs )
        label = np.array(self.data[idx]['y']).astype(int)

        return imgs,label

class FEMNIST(Dataset):
    """ FEMNIST dataset."""

    def __init__(self, root_dir, train=True, maxn=None, same_cls=False, transform=None):
        self.maxn = maxn
        self.same_cls = same_cls
        self.transform = transform
        self.root_dir = root_dir
        self.dataset_name = 'femnist'
        train_data_dir = os.path.join(self.root_dir, self.dataset_name, 'data', 'train')
        test_data_dir = os.path.join(self.root_dir, self.dataset_name, 'data', 'test')
        _, _, train_data, test_data = read_data(train_data_dir, test_data_dir)
       
        if train:
            self.data_dir = train_data_dir
            self.data = train_data
        else:
            self.data_dir = test_data_dir
            self.data = test_data
        self.data_keys = list(self.data.keys())
        self.data_keys.sort()

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        idx = self.data_keys[idx]
        imgs = np.array(self.data[idx]['x']).reshape(-1,28,28)

        if self.transform:
            imgs_new = []
            for img in imgs:
                img = self.transform(img)
                imgs_new.append( img )
            imgs = np.array( imgs_new )
        
        label = np.array(self.data[idx]['y']).astype(int)
        if self.same_cls:
            top_label = np.bincount(label).argmax()
            imgs = imgs[label==top_label]
            label = label[label==top_label]
        if self.maxn and self.maxn < imgs.shape[0]:
            imgs = imgs[ : self.maxn]
            label = label[ : self.maxn]
        return imgs,label

