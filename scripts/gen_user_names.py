from typing import Literal
import pandas as pd
import os
import json
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, Dataset as TorchDataset
import final


def get_user_names():

    if os.path.isfile('dataset/user_names.json'):
        with open('dataset/user_names.json', 'r', encoding='utf8') as fin:
            return {int(uuid): name for uuid, name in json.load(fin).items()}
    with open('name', 'r') as fin:
        names = fin.read()
        names = np.array(list({name.strip() for name in names.split(',')}))
    merged_df = final.data.merged_df(auto_split=False)
    user_order_count = merged_df['uuid_ind'].value_counts()
    user_names: dict[int, str] = {}

    names_for_frequent_user = np.random.choice(
        range(len(names)), (user_order_count > 2).sum(), replace=False
    )
    for i, uid in zip(
        names_for_frequent_user, user_order_count[user_order_count > 2].index
    ):
        user_names[uid] = names[i]

    mask = np.ones_like(names, dtype=bool)
    mask[names_for_frequent_user] = False
    names = names[mask]

    for uid in user_order_count[user_order_count == 2].index:
        user_names[uid] = np.random.choice(names, 1)[0]

    for uid in user_order_count[user_order_count == 1].index:
        user_names[uid] = 'First time buyer'
    # print()
    # print(len(names))
    print(len(user_names))
    with open('dataset/user_names.json', 'w', encoding='utf8') as fout:
        json.dump(user_names, fout)
    return user_names


def main():
    print(get_user_names())


main()
