import itertools
import os
import shutil

import torch
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from config import opt
from dataset import MyDataset, collate_fn
from eval import evalrank
from loss import MyContrastiveLoss
from model import MyModelAll
from utils import Utils
from utils import logger


def main():
    # 获取当前时间
    str_time = Utils.get_time()
    print(str_time)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        "##########################################" + str_time + "###################################################")

    # 定义batch_size等参数-------------------------------------------------------------------------------
    batch_size_train = opt.batch_size_train  # 训练批大小
    data_path = opt.data_path  # 数据集路径
    data_name = opt.data_name  # 数据集名称
    # -------------------------------------------------------------------------------------------------
    # 定义数据预处理

    # 数据集路径
    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)

    # 定义训练数据集
    train_dataset = MyDataset(data_path, data_name, "train", train=True)
    train_num = len(train_dataset)
    logger.info(f"train_dataset_len:{train_num}")

    # 定义测试数据集
    validate_dataset = MyDataset(data_path, data_name, "dev", train=False)  # 若使用5-fold 1K,split应为：testall
    val_num = len(validate_dataset)
    logger.info(f"val_dataset_len:{val_num}")

    # 加载训练数据
    logger.info("开始加载训练数据")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size_train, shuffle=True, pin_memory=True,
                                               collate_fn=collate_fn, drop_last=True)
    logger.info("训练数据加载完毕")

    # 加载测试数据
    logger.info("开始加载测试数据")
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size_train, shuffle=True, pin_memory=True,
                                                  collate_fn=collate_fn, drop_last=True)
    logger.info("测试数据加载完毕")

    # 一些参数--------------------------------------------------------------------------------------
    epochs = opt.epochs  # 训练轮数
    lr_schedules = opt.lr_schedules  # 当epoch=10时，降低学习率

    vse_mean_warmup_epochs = 1

    save_path = opt.save_path  # 模型保存路径
    saved_model_state = './run/' + data_name + ''  # 模型保存路径，如有
    resume = opt.resume  # 是否继续训练

    best_rsum = 0  # 最好检索recall
    margin = opt.margin  # 三元组损失边界值
    learning_rate = opt.learning_rate  # 初始学习率
    max_violation = opt.max_violation  # 损失函数中，使用最大惩罚而不是总数惩罚，后续改变
    rsum_list = []  # 记录结果
    # ---------------------------------------------------------------------------------------------

    # 定义网络，迁移到GPU
    net = MyModelAll()
    net.to(device)
    logger.info("《——模型结构——》")
    logger.info(net.show_model())

    # 定义损失函数，迁移到GPU
    loss_function = MyContrastiveLoss(device=device, margin=margin, max_violation=max_violation).to(device)

    # 冻结某些层
    # for name, param in net.model_txt.named_parameters():
    #     if "bert_basemodel" in name:
    #         param.requires_grad = False

    # 获取网络的所有参数
    net_params = itertools.chain(net.model_img.parameters(), net.model_txt.parameters())
    # 定义优化器
    all_text_params = list(net.model_txt.parameters())
    bert_params = list(net.model_txt.bert_basemodel.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = list()
    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)
    optimizer = torch.optim.AdamW([
        {'params': text_params_no_bert, 'lr': learning_rate},
        {'params': bert_params, 'lr': learning_rate * 2 / 5},
        {'params': net.model_img.parameters(), 'lr': learning_rate},
    ],
        lr=learning_rate, weight_decay=1e-4)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net_params), lr=learning_rate, weight_decay=1e-4)

    # 使用tensorboard记录实验数据
    writer = SummaryWriter(f'logs/{str_time}_{data_name}_epochs{epochs}_batch_size{batch_size_train}')

    logger.info("-------------------训练开始------------------------")
    logger.info("设备：{}".format(device))
    # 开始轮次
    start_epoch = 0
    # 查找是否有检查点，加载检查点
    if resume:
        if os.path.isfile(saved_model_state):
            print("=> loading checkpoint '{}'".format(saved_model_state))
            checkpoint = torch.load(saved_model_state)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            net.load_state_dict_(checkpoint['model'])
            net.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(saved_model_state, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(saved_model_state))

    # 开始训练
    for epoch in range(epochs):
        # 设置模型为训练模式
        net.train()
        # 记录每一轮的损失
        running_loss = 0.0
        logger.info(f"-----当前epoch：{epoch}-----")
        # 调整学习率
        # lr = learning_rate * (0.1 ** (epoch // 5))
        if epoch in lr_schedules:
            logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 0.1
                param_group['lr'] = new_lr
                logger.info('new lr {}'.format(new_lr))
        # 调整最大聚合
        # if epoch >= vse_mean_warmup_epochs:
        #     max_violation = True
        #     loss_function.max_violation_on()
        # model.set_max_violation(opt.max_violation)

        # 开始
        for i, train_data in enumerate(train_loader):
            print("训练进行到batch{}".format(i))
            images, img_len, captions, txt_lengths, _ = train_data
            # 将每一批的图片、标签迁移到GPU上
            images_gpu, captions_gpu = images.to(device), captions.to(device)
            img_lengths = img_len.to(device)
            txt_lengths = torch.tensor(txt_lengths).to(device)
            # pads_gpu = pads.to(device)

            # 五步，完成一次训练，参数更新一次
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播
            scores = net.forward(images_gpu, img_lengths, captions_gpu, txt_lengths)
            # 计算损失函数
            loss = loss_function(scores)
            # 反向传播
            loss.backward()
            # 梯度裁剪

            clip_grad_norm_(net_params, 2.)
            # 优化器更新参数
            optimizer.step()
            # 累加每一个batch的损失，得到一个epoch的总损失
            running_loss += loss.item()
            # tensorboard记录数据
            writer.add_scalars("loss", {f"epoch_{epoch}": loss.item()}, i)
            # logger记录数据
            if i % 20 == 0:
                logger.info(
                    f"train:batch_size:{batch_size_train}当前batch数：{i},epoch:{epoch},当前loss:{loss:.3f}")

        # 记录数据
        writer.add_scalar("总体loss", running_loss, epoch)
        logger.info(f"-----该epoch的总体loss：{running_loss}-----")

        # 每训练一个epoch，进行一次评估
        logger.info(f"----第{epoch}个epoch------评估开始--------")
        rsum = evalrank(net, validate_loader, opt, logger, writer, epoch, max_violation=max_violation,
                        fold5=False)  # 交叉验证选项 fold5=true
        rsum_list.append(rsum)

        # 检测是否达到最好结果
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        # 保存当前结果，最好结果
        save_checkpoint({'epoch': epoch,
                         'model': net.state_dict(),
                         'best_rsum': best_rsum,
                         'Eiters': net.Eiters, }, is_best, prefix='./run/' + data_name + '/',
                        filename=f'checkpoint_{epoch}.pth')

    logger.info("-------------------训练结束------------------------\n\n")
    # print(rsum_list)
    logger.info('最后结果：{}'.format(rsum_list))
    writer.close()


def save_checkpoint(state, is_best, prefix, filename):
    tries = 15
    error = None
    filepath = prefix + filename
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, filepath)
            if is_best:
                shutil.copyfile(filepath, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


if __name__ == '__main__':
    main()
