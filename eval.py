import os
import sys
from typing import Type

# from train import logger
import numpy as np
import torch

from config import opt
from loss import MyContrastiveLoss
from utils import logger


def evalrank(model, data_loader, opt_, mylog, writer, epoch, split='dev', fold5=False, max_violation=False):
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

            sims = shard_xattn(model, pool_imgs, img_embs, cap_pool, cap_embs, cap_lens, opt_, shard_size=64)
            r, rt = i2t(img_embs, sims, return_ranks=True)
            ri, rti = t2i(img_embs, sims, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print()
            mylog.info("rsum: %.1f" % rsum)
            mylog.info("Average i2t Recall: %.1f" % ar)
            mylog.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            mylog.info("Average t2i Recall: %.1f" % ari)
            mylog.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
            # 记录数据
            writer.add_scalar("i2t_r1", r[0], epoch)
            writer.add_scalar("i2t_r5", r[1], epoch)
            writer.add_scalar("i2t_r10", r[2], epoch)

            writer.add_scalar("t2i_r1", ri[0], epoch)
            writer.add_scalar("t2i_r5", ri[1], epoch)
            writer.add_scalar("t2i_r10", ri[2], epoch)

            writer.add_scalar("R_SUM", rsum, epoch)

            # message = "split: %s, Image to text: (%.1f, %.1f, %.1f) " % (split, r[0], r[1], r[2])
            # message += "Text to image: (%.1f, %.1f, %.1f) " % (ri[0], ri[1], ri[2])
            # message += "rsum: %.1f\n" % rsum

            if split == "test" or split == "testall":
                # torch.save({'rt': rt, 'rti': rti}, os.path.join(opt.logger_name, 'ranks.pth.tar'))

                # torch.save({"sims_ti": sims_0, "sims_it": sims_1}, os.path.join(opt.logger_name, 'sims_seperate.pth.tar'))
                torch.save(sims, os.path.join(opt_.sim_path, 'sims.pth.tar'))
        else:
            results = []
            selected_indices = torch.arange(0, len(img_embs), 5)
            img_embs = img_embs[selected_indices]
            pool_imgs = pool_imgs[selected_indices]
            for i in range(5):  # 每次取五分之一数据，交叉验证
                img_embs_shard = img_embs[i * 1000:(i + 1) * 1000]
                pool_imgs_shard = pool_imgs[i * 1000:(i + 1) * 1000]

                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                cap_pool_shard = cap_pool[i * 5000:(i + 1) * 5000]
                cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

                
                sims = shard_xattn(model, pool_imgs_shard, img_embs_shard, cap_pool_shard, cap_embs_shard, cap_lens_shard,
                                opt_, shard_size=64)
                r, rt = i2t(img_embs, sims, return_ranks=True)
                ri, rti = t2i(img_embs, sims, return_ranks=True)
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print()
                results += [list(r) + list(ri) + [rsum, ar, ari]]

            print("-----------------------------------")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            mylog.info("rsum: %.1f" % mean_metrics[10])
            mylog.info("Average i2t Recall: %.1f" % mean_metrics[11])
            mylog.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
            mylog.info("Average t2i Recall: %.1f" % mean_metrics[12])
            mylog.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])
            # 记录数据
            writer.add_scalar("i2t_r1", mean_metrics[0], epoch)
            writer.add_scalar("i2t_r5", mean_metrics[1], epoch)
            writer.add_scalar("i2t_r10", mean_metrics[2], epoch)

            writer.add_scalar("t2i_r1", mean_metrics[5], epoch)
            writer.add_scalar("t2i_r5", mean_metrics[6], epoch)
            writer.add_scalar("t2i_r10", mean_metrics[7], epoch)

            writer.add_scalar("R_SUM", mean_metrics[10], epoch)
            rsum = mean_metrics[10]
    return rsum


def encode_data(model, data_loader, max_violation=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 开始取出dataloader中的所有数据保存，
    # 找到每一batch中，文本的最大长度
    max_n_word = 0
    for i, (_, _, _, txt_lengths, _) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(txt_lengths))

    # 定义返回的总体嵌入的数据
    img_embs = None
    cap_embs = None
    cap_lens = None
    # 定义池化的图文
    pool_imgs = None
    pool_txts = None

    # 定义损失函数
    loss_function = MyContrastiveLoss(device=device, margin=opt.margin, max_violation=max_violation).to(device)

    # 初始化
    for i, val_data in enumerate(data_loader):
        images, img_lengths, captions, txt_lengths, indexes = val_data

        # 将每一批的图片、标签迁移到GPU上
        val_images_gpu, val_captions_gpu = images.to(device), captions.to(device)
        txt_lengths = torch.tensor(txt_lengths).to(device)
        img_lengths = img_lengths.to(device)

        # 前向传播
        model.eval()
        img_emb, txt_emb, poolimg, pooltxt, lens = model.forward_emb(val_images_gpu, img_lengths, val_captions_gpu,
                                                                     txt_lengths)
        score = model.forward_score(poolimg, img_emb, pooltxt, txt_emb, lens, opt)
        # txt长度
        # txt_len = txt_emb.size(1)
        lens = torch.tensor(lens)
        # 构建一个批次的相同长度
        # cap_len = torch.full((len(txt_lengths),), txt_len)
        # 初始化数据
        if img_embs is None:
            img_embs = torch.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)), device=device)
            pool_imgs = torch.zeros((len(data_loader.dataset), img_emb.size(2)), device=device)
            cap_embs = torch.zeros((len(data_loader.dataset), max_n_word, txt_emb.size(2)), device=device)
            pool_txts = torch.zeros((len(data_loader.dataset), txt_emb.size(2)), device=device)

            cap_lens = [0] * len(data_loader.dataset)

        # 缓存数据
        indexes = torch.tensor(indexes)
        img_embs[indexes] = img_emb.detach().clone()
        cap_embs[indexes, :max(txt_lengths), :] = txt_emb.detach().clone()
        for j, nid in enumerate(indexes):
            cap_lens[nid] = lens[j].item()
        pooled_img, pooled_txt = poolimg, pooltxt
        pool_imgs[indexes] = pooled_img
        pool_txts[indexes] = pooled_txt

        # measure accuracy and record loss
        loss = loss_function(score)
        logger.info(f"测试：第{i}batch，共{len(data_loader)}batch size，当前batch的loss：{loss}")

        del images, captions
    return img_embs, cap_embs, pool_imgs, pool_txts, cap_lens


def shard_xattn_Full(model, images_fc, images, caption_ht, captions, caplens, opt: Type[opt], shard_size):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d_t2i = torch.zeros((len(images), len(captions))).cuda()
    d_i2t = torch.zeros((len(images), len(captions))).cuda()

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn: batch (%d,%d)' % (i, j))

            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im_fc = images_fc[im_start:im_end]
            im_emb = images[im_start:im_end]
            h = caption_ht[cap_start:cap_end]
            s = captions[cap_start:cap_end]
            l = caplens[cap_start:cap_end]
            sim_list_t2i = model.xattn_score_Text_(im_fc, im_emb, h, s, l, opt)
            sim_list_i2t = model.xattn_score_Image_(im_fc, im_emb, h, s, l, opt)
            # assert len(sim_list_t2i) == opt.iteration_step and len(sim_list_i2t) == opt.iteration_step
            # for k in range(opt.iteration_step):
            d_t2i[im_start:im_end, cap_start:cap_end] = sim_list_t2i.data
            d_i2t[im_start:im_end, cap_start:cap_end] = sim_list_i2t.data

    # score = 0
    # for j in range(opt.iteration_step):
    #     score += d_t2i[j]
    # for j in range(opt.iteration_step):
    #     score += d_i2t[j]
    score = d_i2t + d_i2t
    return score


def shard_xattn_Image(model, images_fc, images, caption_ht, captions, caplens, opt: Type[opt], shard_size):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = torch.zeros((len(images), len(captions))).cuda()

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn: batch (%d,%d)' % (i, j))

            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im_fc = images_fc[im_start:im_end]
            im_emb = images[im_start:im_end]
            h = caption_ht[cap_start:cap_end]
            s = captions[cap_start:cap_end]
            l = caplens[cap_start:cap_end]
            sim_list = model.xattn_score_Image_(im_fc, im_emb, h, s, l, opt)
            # assert len(sim_list) == opt.iteration_step

            d[im_start:im_end, cap_start:cap_end] = sim_list.data

    score = d
    return score


def shard_xattn_Text(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images) - 1) / shard_size) + 1
    n_cap_shard = int((len(captions) - 1) / shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = torch.zeros((len(images), len(captions))).cuda()

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn: batch (%d,%d)' % (i, j))

            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im_fc = images_fc[im_start:im_end]
            im_emb = images[im_start:im_end]
            h = caption_ht[cap_start:cap_end]
            s = captions[cap_start:cap_end]
            l = caplens[cap_start:cap_end]
            sim_list = model.xattn_score_Text_(im_fc, im_emb, h, s, l, opt)
            # assert len(sim_list) == opt.iteration_step
            # for k in range(opt.iteration_step):
            #     if len(sim_list[k]) != 0:
            d[im_start:im_end, cap_start:cap_end] = sim_list.data

    score = d
    return score


def shard_xattn(model, img_pool, images, txt_pool, captions, caplens, opt: Type[opt], shard_size=64):
    if opt.model_mode == "full":

        sims = shard_xattn_Full(model, img_pool, images, txt_pool, captions, caplens, opt, shard_size=128)
    elif opt.model_mode == "image":
        sims = shard_xattn_Image(model, img_pool, images, txt_pool, captions, caplens, opt, shard_size=128)
        sims2 = shard_xattn_t2i_i2t(images, captions, img_pool, txt_pool)
        sims = sims + sims2

    elif opt.model_mode == "text":
        sims = shard_xattn_Text(model, img_pool, images, txt_pool, captions, caplens, opt, shard_size=128)
        sims2 = shard_xattn_t2i_i2t(images, captions, img_pool, txt_pool)
        sims = sims + sims2

    else:
        sims = shard_xattn_t2i_i2t(images, captions, img_pool, txt_pool)
    return sims


# 计算整个测试数据集的相似度矩阵，分批次
def shard_xattn_t2i_i2t(images, captions, pool_imgs, pool_texts, device="cuda:0", shard_size=128):
    # 图像部分切片数量
    n_im_shard = (len(images) - 1) // shard_size + 1
    print("img shard num:{}".format(n_im_shard))
    # 文本部分切片数量
    n_cap_shard = (len(captions) - 1) // shard_size + 1
    print("text shard num:{}".format(n_cap_shard))

    # 返回的相似度矩阵
    d = torch.zeros((len(images), len(captions))).to(device=device)
    # 图像区域
    img_region = images.size(1)

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        tmp_img_size = im_end - im_start

        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn: batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            tmp_txt_size = cap_end - cap_start

            img = images[im_start:im_end].to(device)

            # 获取对应的切片的池化后的图像文本
            pooled_img = pool_imgs[im_start: im_end]
            pooled_txt = pool_texts[cap_start: cap_end]

            # 计算s1

            sim = get_sim(pooled_img, pooled_txt).cpu()
            # 得到s1+s2
            scores = sim
            # 将切片的相似度矩阵放入整体矩阵中
            d[im_start:im_end, cap_start:cap_end] = scores
    sys.stdout.write('结束！！！！！！\n')
    return d


def i2t(images, sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """

    sims = sims.cpu().numpy()
    npts = images.size(0)
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 倒序
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr


def t2i(images, sims, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    sims = sims.cpu().numpy()

    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10, medr, meanr


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities
