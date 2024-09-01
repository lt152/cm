import os
import random

import numpy as np
import torch
from torch.utils import data

from model import tokenizer


class MyDataset(data.Dataset):
    def __init__(self, data_path, data_name, data_split, train):
        self.train = train
        # 设置数据路径
        # data_path = "./data"
        data_path = os.path.join(data_path, data_name)
        self.captions = []  # 145000
        # 打开数据
        with open(data_path + "/" + f"{data_split}_caps.txt", "rb") as f:
            for line in f:
                # 去除行尾的换行符，并将处理后的标题添加到标题列表中
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(data_path + "/" + f'{data_split}_ims.npy')
        # 获取标题的数量，并赋值给类的属性length
        self.length = len(self.captions)
        # 除以5
        if self.images.shape[0] != self.length:
            # 将类的属性im_div设置为5
            self.im_div = 5
        else:
            # 否则，将类的属性im_div设置为1
            self.im_div = 1
        # Image features
        # self.images = np.load(data_path + "/" + f'{data_split}_ims.npy')
        # self.mmap_arr = np.memmap(path, dtype=np.float32, mode='r', shape=(113287, 36, 2048))

    def __getitem__(self, index):

        # handle the image redundancy
        img_id = index // self.im_div
        # 获取图像数据
        image = self.images[img_id]

        # if self.train:  # Size augmentation for region feature
        #     num_features = image.shape[0]
        #     rand_list = np.random.rand(num_features)
        #     image = image[np.where(rand_list > 0.20)]
        image = torch.tensor(image)

        # 获取对应的标题
        caption = self.captions[index]
        caption = str(caption).lower()
        # 使用bert token 符号化
        tokenized = tokenizer(caption)
        input_ids = tokenized['input_ids']
        caption = torch.tensor(input_ids)

        return image, caption, index, img_id

    def __len__(self):
        return self.length


# 每一个文本长度不一样，因此需要预处理，在dataloader处调用
def collate_fn(data_):
    # Sort a data list by caption length
    data_.sort(key=lambda x: len(x[1]), reverse=True)

    # images:bs个（36,2048）
    # captions:bs个
    # [a,b,c],
    # [a,b,c,d],
    # [a,b],...
    # indexes: 文字索引[]
    # img_ids：图片索引[]
    images, captions, indexes, img_ids = zip(*data_)
    ###########################################################
    img_lengths = [len(image) for image in images]
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]
    img_lengths = torch.Tensor(img_lengths)
    ############################################################
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    # images = torch.stack(images, 0)  # 多个图片连在一起

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]  # [15,14,13,12,11,10,9,8,7] 句子长度
    # 一个batch_size 内文本最大长度
    targets = torch.zeros(len(captions), max(lengths)).long()
    # pad = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        # pad[i, :end] = 1
    # for i in enumerate(captions):

    # images:bs,36,2048
    # captions:补零
    # [a,b,c,0],
    # [a,b,c,d],
    # [a,b,0,0],...
    # indexes: 文字索引
    # img_ids：图片索引
    return all_images, img_lengths, targets, lengths, indexes


def process_caption(tokenizer, tokens, train=True):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target
