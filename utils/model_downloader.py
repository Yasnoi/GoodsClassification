import os
from transformers import AutoModel, AutoTokenizer


if __name__ == '__main__':
    os.environ['http_proxy'] = 'http://127.0.0.1:7897'
    os.environ['https_proxy'] = 'http://127.0.0.1:7897'

    model_name = 'distilbert-base-uncased'
    save_dir = 'data/'

    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
