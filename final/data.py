from typing import Literal
import requests
import pandas as pd
import json
import os
from urllib.parse import urlparse
import hashlib
from functools import lru_cache

ORDERS_SPLIT_JSON = 'https://www.dropbox.com/scl/fi/evqdtwkcwf8ri51x1tx1x/orders_split.json?rlkey=z67954kc1tsv4lyrmnpfy8ewq&st=f2jeijkj&dl=0'
# AVAILABLE_PRODUCTS_JSON = 'https://www.dropbox.com/scl/fi/g6218d4m6t27rkfxrd9jv/available_products.json?rlkey=lmv00kjrf08mt01gq8d5xpw9h&st=jeamothx&dl=0'
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
N_RPODUCTS = 357
AVAILABLE_PRODUCT_IDS = [5, 23, 24, 25, 28, 46, 51, 56, 76, 77, 79, 81, 87, 93, 95, 119, 133, 151, 173, 183, 189, 195, 212, 232, 236, 264, 282, 295, 320, 324, 345, 352, 353, 365, 373, 376, 380, 381, 383, 386, 387, 400, 401, 437, 449, 454, 455, 457, 459, 477, 499, 520, 526, 527, 530, 531, 534, 535, 554, 556, 565, 578, 579, 592, 596, 636, 644, 645, 646, 647, 649, 695, 701, 736, 737, 740, 753, 765, 816, 818, 824, 835, 836, 890, 891, 925, 926, 949, 1014, 1026, 1034, 1036, 1037, 1039, 1065, 1075, 1098, 1099, 1100, 1121, 1123, 1125, 1147, 1195, 1197, 1198, 1210, 1216, 1217, 1232, 1233, 1234, 1235, 1236, 1237, 1240, 1242, 1285, 1404, 1418, 1419, 1437, 1440, 1441, 1442, 1452, 1453, 1454, 1455, 1491, 1494, 1495, 1496, 1497, 1499, 1512, 1515, 1516, 1539, 1575, 1686, 1687, 1688, 1700, 1701, 1719, 1722, 1729, 1746, 1758, 1767, 1799, 1816, 1857, 1858, 1859, 1860, 1861, 1864, 1893, 1906, 1914, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1997, 1999, 2001, 2005, 2019, 2027, 2048, 2049, 2077, 2083, 2124, 2125, 2239, 2240, 2254, 2265, 2266, 2270, 2312, 2477, 2495, 2570, 2572, 2574, 2576, 2577, 2678, 2679, 2712, 2782, 2799, 2803, 2823, 2832, 2840, 2851, 2886, 2887, 2903, 2907, 2917, 2918, 2947, 2955, 2988, 3021, 3039, 3062, 3227, 3232, 3233, 3239, 3304, 3306, 3319, 3329, 3340, 3346, 3362, 3366, 3367, 3369, 3373, 3395, 3400, 3409, 3427, 3431, 3489, 3521, 3522, 3523, 3524, 3525, 3530, 3532, 3534, 3536, 3537, 3538, 3539, 3540, 3542, 3543, 3545, 3546, 3547, 3549, 3551, 3568, 3650, 3663, 3680, 3728, 3794, 3798, 3812, 3818, 3819, 3820, 3839, 3843, 3864, 3883, 3901, 3915, 3932, 3943, 3972, 3976, 4001, 4026, 4031, 4036, 4043, 4045, 4047, 4054, 4055, 4056, 4059, 4060, 4070, 4075, 4083, 4092, 4102, 4103, 4125, 4139, 4140, 4141, 4144, 4180, 4181, 4183, 4194, 4196, 4213, 4214, 4216, 4227, 4230, 4231, 4232, 4255, 4261, 4267, 4293, 4302, 4322, 4345, 4356, 4413, 4422, 4473, 4475, 4478, 4479, 4489, 4517, 4518, 4526, 4536, 4538, 4540, 4545, 4554, 4558, 4580, 4589, 4591, 4593, 4594, 4628, 4631, 4634, 4635, 4636, 4648, 4649, 4650, 4651, 4652, 4670, 4675, 4705] # yapf: disable

# def available_product_ids(cache_dir: str = CACHE_DIR):
#     ids = _download(AVAILABLE_PRODUCTS_JSON, cache_dir)
#     assert len(ids) == N_RPODUCTS
#     return ids


def load_df(cache_dir: str = CACHE_DIR):
    return _download(LOAD_PKL, cache_dir)


def _create_split(
    merged_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
):

    total_time_interval = merged_df['timestamp'].max(
    ) - merged_df['timestamp'].min()
    train_stamp = merged_df['timestamp'].min() + int(
        total_time_interval * train_ratio
    )
    val_stamp = merged_df['timestamp'].min() + int(
        total_time_interval * (train_ratio + val_ratio)
    )

    train_idx = merged_df['timestamp'] <= train_stamp
    val_idx = (~train_idx) & (merged_df['timestamp'] <= val_stamp)
    test_idx = merged_df['timestamp'] > val_stamp

    return {
        'train': merged_df[train_idx],  #.reset_index(drop=True),
        'val': merged_df[val_idx],  #.reset_index(drop=True),
        'test': merged_df[test_idx],  #.reset_index(drop=True),
    }


def merged_df(
    cache_dir: str = CACHE_DIR
) -> dict[Literal['train', 'val', 'test'], pd.DataFrame]:
    merged_df: pd.DataFrame = _download(MERGED_PKL, cache_dir)
    return _create_split(merged_df)
    # order_splits = _download(ORDERS_SPLIT_JSON, cache_dir)
    # merged_df = merged_df.set_index('order_id')
    # return {
    #     split: merged_df.loc[order_split]
    #     for split, order_split in order_splits.items()
    # }
