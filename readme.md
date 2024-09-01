# 基于交叉注意力与特征聚合的跨模态图文检索研究

- 数据集：https://www.kaggle.com/datasets/kuanghueilee/scan-features
- bert模型：https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin?download=true
- 参数配置文件：config.py
- 安装环境
```bash
conda env create -n crossmodal -f environment.yml
```
* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```
- 开始训练
```bash
python train.py
```



