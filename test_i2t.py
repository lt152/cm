import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import opt
from dataset import MyDataset, collate_fn
from eval import i2t, t2i, encode_data

from model import MyModelAll
from utils import Utils, logger

'''
可视化实验
'''


def main():
    str_time = Utils.get_time()
    print(str_time)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        "##########################################" + str_time + "###################################################")

    str_model_path = "C:/AMyFile/实验数据/f30k model/model_best_f30k.pth"

    assert os.path.exists(str_model_path), "{} path does not exist.".format(str_model_path)
    img_path = ""
    txt = ""
    ###########################################################################
    # 定义网络，迁移到GPU
    net = MyModelAll()
    net.to(device)
    logger.info("《——模型结构——》")
    logger.info(net.show_model())

    # if os.path.isfile(str_model_path):
    #     print("=> loading checkpoint '{}'".format(str_model_path))
    #     checkpoint = torch.load(str_model_path)
    #     # logger.info(checkpoint)
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_rsum = checkpoint['best_rsum']
    #     net.load_state_dict(checkpoint['model'])
    #     net.Eiters = checkpoint['Eiters']
    #     print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
    #           .format(str_model_path, start_epoch, best_rsum))
    # else:
    #     print("=> no checkpoint found at '{}'".format(str_model_path))
    loc = 'D:/xunlei_download/archive/data/data/f30k_precomp/'
    data_split = 'test'
    # images_matrix = np.load(loc + '{}_ims.npy'.format(data_split))

    data_path = "D:/experiment/project_experiment/data"  # 数据集路径
    data_name = "f30k_precomp"  # 数据集名称
    batch_size = 64

    validate_dataset = MyDataset(data_path, data_name, "test", train=False)

    # 加载测试数据
    logger.info("开始加载测试数据")
    test_loader = DataLoader(validate_dataset, batch_size, shuffle=False, pin_memory=True,
                             collate_fn=collate_fn, drop_last=False)
    logger.info("测试数据加载完毕")

    sims = evalrank(net, test_loader, opt, logger)
    sims = sims.cpu()
    sims = sims.numpy()
    np.save('./sims.npy', sims)
    ##################################################################
    sims = np.load('./sims.npy')
    # sims = np.load("")
    # exit(0)

    randint = random.randint(0, 999)
    index = randint
    # index = 660
    print(randint)
    inds = np.argsort(sims[index])[::-1]  # 倒序
    my_data_test = inds[:5]
    resu = sims[index, my_data_test]
    my_data_should = [5 * index + i for i in range(5)]
    print("前五测试排名：", my_data_test)
    print("前五测试相似度结果：", resu)
    # print("前五应有数据：", my_data_should)
    # Score

    sims1 = np.load('./sims_scan.npy')  # 基线模型
    inds1 = np.argsort(sims1[index])[::-1]  # 倒序
    my_data_test1 = inds1[:5]
    resu1 = sims1[index, my_data_test1]
    my_data_should1 = [5 * index + i for i in range(5)]
    print()
    print("前五测试排名：", my_data_test1)
    print("前五测试相似度结果：", resu1)
    print()
    print("前五应有数据：", my_data_should1)
    # rank = 1e20
    # # x = np.zeros(5)
    # for i in range(5 * index, 5 * index + 5, 1):
    #     tmp = np.where(inds == i)[0][0]
    #     if tmp < rank:
    #         rank = tmp


def evalrank(model, data_loader, opt_, mylog, fold5=False, max_violation=True):
    mylog.info("-------- evaluation --------")
    model.eval()
    with torch.no_grad():
        img_embs, cap_embs, pool_imgs, cap_pool, cap_lens = encode_data(model, data_loader, max_violation=max_violation)
        if not fold5:
            # no cross-validation, full evaluation
            # img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
            # 图像去除冗余
            selected_indices = torch.arange(0, len(img_embs), 5)
            img_embs = img_embs[selected_indices]
            pool_imgs = pool_imgs[selected_indices]

            from eval import shard_xattn
            sims = shard_xattn(model, pool_imgs, img_embs, cap_pool, cap_embs, cap_lens, opt_, shard_size=64)

    return sims


if __name__ == '__main__':
    main()
