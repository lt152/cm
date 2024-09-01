import torch


class opt:
    lambda_lse = 6.0
    lambda_softmax = 9.0
    # start_epoch = 0
    # 选择消融实验方式
    model_mode = 'full'  # image,text,full，或在model.py中修改
    batch_size_train = 64  # 训练批大小
    epochs = 20  # 训练轮数
    lr_schedules = [10, 15]
    data_path = "/clusters/data_4090/user_data/liteng/projectList/dataset/data/data"  # 数据集路径,
    # data_path = r"D:\xunlei_download\archive\data\data"  # 数据集路径
    data_name = "f30k_precomp"  # 数据集名称
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resume = False  # 是否继续训练
    save_path = './checkpoint'  # 模型保存路径
    saved_model_state = './checkpoint/checkpoint_5.pth'  # 模型
    margin = 0.3  # 三元组损失边界值
    learning_rate = 0.0005  # 初始学习率
    max_violation = True  # 损失函数中，使用最大而不是总数
    iteration_step = 2
    sim_path = './simpath'
