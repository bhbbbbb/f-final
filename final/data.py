import requests
import pandas as pd
import json
import os
from urllib.parse import urlparse
import hashlib
from functools import lru_cache

ORDERS_SPLIT_JSON = 'https://www.dropbox.com/scl/fi/evqdtwkcwf8ri51x1tx1x/orders_split.json?rlkey=z67954kc1tsv4lyrmnpfy8ewq&st=f2jeijkj&dl=0'
AVAILABLE_PRODUCTS_JSON = 'https://www.dropbox.com/scl/fi/g6218d4m6t27rkfxrd9jv/available_products.json?rlkey=lmv00kjrf08mt01gq8d5xpw9h&st=jeamothx&dl=0'
LOAD_PKL = 'https://www.dropbox.com/scl/fi/9gwtyguosa7kcbp2ltd20/load_df_v2.pkl?rlkey=7j3w0vztmnkqjrdc6pw31kxoq&st=fh4u9b17&dl=0'
MERGED_PKL = 'https://www.dropbox.com/scl/fi/8cj84noi9pzr399r35t6y/merged_df_v2.pkl?rlkey=hgyzz75nwpch9zo1shpxehk2k&st=rd25pwa6&dl=0'


@lru_cache(maxsize=4)
def _download(url: str, cache_dir: str):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    file_name = os.path.basename(urlparse(url).path)
    ext = os.path.splitext(file_name)[-1]
    assert ext in ('.json', '.pkl')
    file_path = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest())

    if not os.path.exists(file_path):
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        res = requests.get(url, headers=headers)
        assert res.status_code == 200
        with open(file_path, 'wb') as fout:
            fout.write(res.content)

    if ext == '.json':
        with open(file_path, 'r', encoding='utf8') as fin:
            return json.load(fin)
    else:
        assert ext == '.pkl'
        return pd.read_pickle(file_path)


CACHE_DIR = 'dataset'


def available_product_ids(cache_dir: str = CACHE_DIR):
    return _download(AVAILABLE_PRODUCTS_JSON, cache_dir)


def load_df(cache_dir: str = CACHE_DIR):
    return _download(LOAD_PKL, cache_dir)


def merged_df(cache_dir: str = CACHE_DIR):
    merged_df: pd.DataFrame = _download(MERGED_PKL, cache_dir)
    order_splits = _download(ORDERS_SPLIT_JSON, cache_dir)
    merged_df = merged_df.set_index('order_id')

    return {
        split: merged_df.loc[order_split]
        for split, order_split in order_splits.items()
    }
