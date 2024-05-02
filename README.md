# Neural machine translation English-Vietnamese with Transformer

<a target="_blank" href="https://githubtocolab.com/minhnguyent546/nmt-en-vi/blob/master/nmt-en-vi-colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
</a>
<a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/minhnguyent546/nmt-en-vi/blob/master/nmt-en-vi-kaggle.ipynb">
  <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle">
</a>

A Transformer implementation from scratch, crafted following the principles outlined in the paper 'Attention Is All You Need', serves as a robust solution for tasks such as neural machine translation and grammatical error correction.

## Setup

- Clone this repo:
```bash
git clone https://github.com/minhnguyent546/nmt-en-vi.git
cd nmt-en-vi
```

- Install required dependencies:
```bash
pip install -r requirements.txt
```

## Training and testing

Please take a look at the config file located at `config/config.yaml`, and change the `train`, `test`, and `validation` paths to your local files.

- Preprocessing the data:
```bash
python preprocess_nmt.py --config 'config/config.yaml'
```

- To train the model:
```bash
python train_nmt.py --config 'config/config.yaml'
```

- To test the model:
```bash
python test_nmt.py --config 'config/config.yaml'
```
