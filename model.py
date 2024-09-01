from typing import Type

import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch.nn.functional as F
from config import opt
from MyPool import MyPool
from MLP import MLP

bert_model_path = "./model/"
tokenizer = BertTokenizer.from_pretrained(bert_model_path)  # 通过词典导入分词器
model_config = BertConfig.from_pretrained(bert_model_path)  # 导入配置文件


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    tmp = (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    return tmp


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


def func_attention(query, context, smooth, eps=1e-8, raw_feature_norm="clipped_l2norm"):
    """
    query: (n_context, queryL, d)cap text
    context: (n_context, sourceL, d) image
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)

    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type.")
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=-1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


# 定义模型
class MyModelAll(nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()
        self.grad_clip = 2.0  # 梯度裁剪
        self.opt = opt
        self.device = opt.device

        # 构建模型
        self.model_img = MyImgModel(no_norm=False)
        self.model_txt = MyBert(no_txtnorm=False)
        # self.model_gpo = MyGPO(device=device, in_dim=2, out_dim=1)
        # 损失函数
        # self.contrastive_loss = MyContrastiveLoss(margin=0.2, max_violation=False)
        self.linear_t2i = nn.Linear(embed_size * 2, embed_size)
        # self.gate_t2i = nn.Linear(embed_size * 2, embed_size)
        self.linear_i2t = nn.Linear(embed_size * 2, embed_size)
        # self.gate_i2t = nn.Linear(embed_size * 2, embed_size)
        self.Eiters = 0

    def forward_emb(self, images, img_len, captions, lengths, ):
        self.Eiters += 1
        # 前向传播

        img_emb, pooled_img = self.model_img(images, img_len)
        txt_emb, pooled_txt = self.model_txt(captions, lengths)
        # 学习池化参数
        n = txt_emb.size(1)  # 每个batch的文本长度已经一样了
        lens = [n] * txt_emb.size(0)
        return img_emb, txt_emb, pooled_img, pooled_txt, lens

    def forward_score(self, pool_img, img_emb, pool_txt, cap_emb, cap_len, opt: Type[opt]):
        score = None
        if opt.model_mode == 'full':
            # s1+s2
            scores_t2i = self.xattn_score_Text_(pool_img, img_emb, pool_txt, cap_emb, cap_len, self.opt)
            scores_i2t = self.xattn_score_Image_(pool_img, img_emb, pool_txt, cap_emb, cap_len, self.opt)

            # 这里控制是否加入S3
            score = get_sim(pool_img, pool_txt)
            score = scores_t2i + scores_i2t + score

        elif opt.model_mode == 'image':
            # s1+s3
            scores_i2t = self.xattn_score_Image_(pool_img, img_emb, pool_txt, cap_emb, cap_len, self.opt)
            score = get_sim(pool_img, pool_txt)

            score = scores_i2t + score
        elif opt.model_mode == 'text':
            # s2+s3
            scores_t2i = self.xattn_score_Text_(pool_img, img_emb, pool_txt, cap_emb, cap_len, self.opt)
            score = get_sim(pool_img, pool_txt)

            score = scores_t2i + score
        else:
            # s3
            score = get_sim(pool_img, pool_txt).to(self.device)
        return score

    def forward(self, images, img_len, captions, lengths, ):
        img_emb, txt_emb, pooled_img, pooled_txt, lens = self.forward_emb(images, img_len, captions, lengths)
        scores = self.forward_score(pooled_img, img_emb, pooled_txt, txt_emb, lens, self.opt)
        # theta, theta_ = self.model_gpo(img_emb, txt_emb)
        return scores

    def to_device(self, device):
        self.model_img.to(device)
        self.model_txt.to(device)
        self.to_device(device)
        # self.model_gpo.to(device)

    def set_train(self):
        self.model_img.train()
        self.model_txt.train()
        self.train()
        # self.model_gpo.train()

    def set_eval(self):
        self.model_img.eval()
        self.model_txt.eval()
        self.eval()
        # self.model_gpo.eval()

    def state_dict_(self):
        state_dict = [self.state_dict_()]
        return state_dict

    def load_state_dict_(self, state_dict):
        self.model_img.load_state_dict(state_dict=state_dict[0])
        self.model_txt.load_state_dict(state_dict=state_dict[1])
        # self.model_gpo.load_state_dict(state_dict=state_dict[2])

    def show_model(self):
        list_ = [self.model_img, self.model_txt]
        return list_

    def xattn_score_Text_(self, img_poo, img_emb, txt_poo, txt_emb, cap_lens, opt: Type[opt]):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        n_image = img_emb.size(0)
        n_caption = txt_emb.size(0)
        images = img_emb.float()
        captions_all = txt_emb.float()
        # caption_ht = txt_poo.float()
        # img_poo = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            query = cap_i_expand
            context = images
            weight = 0

            attn_feat, _ = func_attention(query, context, smooth=opt.lambda_softmax)

            tmp_expand = txt_poo[i].expand(attn_feat.size(0), attn_feat.size(1), -1)

            row_sim = cosine_similarity(tmp_expand, attn_feat, dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)
            # 使用logsumexp池化
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse

            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)

        return similarities

    def xattn_score_Image_(self, img_poo, img_emb, txt_poo, txt_emb, cap_lens, opt: Type[opt]):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = []
        n_image = img_emb.size(0)
        n_caption = txt_emb.size(0)
        n_region = img_emb.size(1)
        img_emb = img_emb.float()
        txt_emb = txt_emb.float()
        txt_poo = txt_poo.float()
        # img_poo = img_emb.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = txt_emb[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_h_i = txt_poo[i].unsqueeze(0).unsqueeze(0).contiguous()
            cap_h_i_expand = cap_h_i.expand_as(img_emb)

            query = img_emb
            # query = img_poo.unsqueeze(1).contiguous()
            context = cap_i_expand
            weight = 0

            attn_feat, _ = func_attention(query, context, smooth=opt.lambda_softmax)

            img_poo_ = img_poo.unsqueeze(1)
            img_poo_ = img_poo_.expand(-1, attn_feat.size(1), -1)

            row_sim = cosine_similarity(img_poo_, attn_feat, dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)
            # 使用logsumexp池化
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim) / opt.lambda_lse

            similarities.append(row_sim)
        similarities = torch.cat(similarities, 1)

        return similarities


class MyImgModel(nn.Module):
    def __init__(self, no_norm=False, weight='xavier', out_features=1024):
        super().__init__()
        self.no_norm = no_norm
        self.fc = nn.Linear(2048, out_features)  # 2048*1024
        # self.batch_norm = nn.BatchNorm1d(1024)
        self.mlp = MLP(2048, out_features // 2, out_features, 2)
        self.gpool = MyPool()
        # self.do = nn.Dropout(p=0.2)

        if weight == 'xavier':
            print("img:xavier")
            nn.init.xavier_normal_(self.fc.weight)  # 权重初始化方式
        elif weight == 'kaiming':
            print("img:kaiming")
            nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x, image_lengths):
        features = self.fc(x)
        features = self.mlp(x) + features
        # features = self.do(features)

        pooled_feature, pool_weights = self.gpool(features, image_lengths)

        # 是否进行图像输出归一化？
        if not self.no_norm:
            features = torch.nn.functional.normalize(features, p=2, dim=-1, eps=1e-8)
            pooled_feature = l2norm(pooled_feature, dim=-1)
            # features=torch.transpose(features,1,2)
            # features = self.batch_norm(features)
            # features=torch.transpose(features,1,2)

        return features, pooled_feature


class MyBert(nn.Module):
    def __init__(self, no_txtnorm=False, weight='xavier', out_features=1024):
        super().__init__()

        self.no_txtnorm = no_txtnorm
        self.bert_basemodel = BertModel.from_pretrained(bert_model_path, config=model_config)
        self.fc = nn.Linear(768, out_features)
        # self.relu = nn.LeakyReLU()
        self.gpool = MyPool()

        if weight == 'xavier':
            print("txt:xavier")
            nn.init.xavier_normal_(self.fc.weight)  # 权重初始化方式
        elif weight == 'kaiming':
            print("txt:kaiming")
            nn.init.kaiming_normal_(self.fc.weight)  # 权重初始化方式

    def forward(self, x, lengths):
        # self.bert_basemodel.eval()

        pad = (x != 0).float()
        bert_output = self.bert_basemodel(x, pad)
        output_ = self.fc(bert_output[0])
        cap_len = lengths
        pooled_features, pool_weights = self.gpool(output_, cap_len)

        if not self.no_txtnorm:
            output_ = nn.functional.normalize(output_, p=2, dim=-1, eps=1e-8)
            pooled_features = l2norm(pooled_features, dim=-1)

        # output_ = self.relu(output_)
        return output_, pooled_features
        # return bert_output[0]
