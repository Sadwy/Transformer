# Install
```shell
# 系统: Ubuntu 21.04
# 创建conda环境
conda create -n transformer python=3.8 -y
conda activate transformer

# 配置环境包
conda install -c pytorch torchtext
pip install pytz six spacy
```
- 上述torchtext包的安装指令可能会导致安装cpu版的torch. 如果安装了cpu版本, 建议下载whl文件本地安装 (百度).
    - cuda_11.3选用的[whl文件](https://download.pytorch.org/whl/): torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl 和 torchtext-0.11.0-cp38-cp38-linux_x86_64.whl

# Preparation
```shell
# 下载预训练模型
# 网络问题可能导致下载失败, 可以多试几次, 或其他方法(比如下载压缩包本地安装, 百度)
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

# Train
```shell
# 训练 (单卡)
python train.py -src_data data/english.txt -trg_data data/french.txt -src_lang en_core_web_sm -trg_lang fr_core_news_sm -epochs 10
```
- 训练结束后会提示是否保存权重, 建议保存到 `weights` 文件夹 (会自动创建).

# Inference
```shell
# 推理准备, 下载wordnet
pip install nltk
mkdir ~/data/corpora && cd ~/data/corpora
wget https://github.com/nltk/nltk_data/raw/gh-pages/packages/corpora/wordnet.zip && unzip wordnet.zip
cd ~ && ln -s ~/data ~/nltk_data

# 推理
python translate.py -load_weights weights -src_lang en_core_web_sm -trg_lang fr_core_news_sm
```
- 支持单句翻译和文件翻译, 按照运行中的文字提示操作. (文字提示的实现代码在translate.py L120-L153)
- 推理指令中的 `weights` 即是训练时保存权重的文件夹.
- 数据集中的部分字符不规范, 导致模型预测结果中可能包含一些字符编码.
    - 例如, 训练集中有些空格是 `\u202f` 和 `\xa0` 类型, 导致部分翻译结果中包含该字符.

## 1. 性能评估
- BLEU评估: 推理时输入 `$config` 设置评估, 默认不评估.
    - 评估函数是translate.py的calculate_bleu函数.
