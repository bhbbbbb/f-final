import re
import os
import json
import pandas as pd
from functools import reduce


def get_unique_product_ids(dataset_root: str = 'dataset'):

    save_path = os.path.join(dataset_root, 'unique_products.json')
    if os.path.isfile(save_path):
        with open(save_path, 'r', encoding='utf8') as fin:
            return json.load(fin)

    load_df: pd.DataFrame = pd.read_pickle(
        os.path.join(dataset_root, 'load_df_ped.pkl')
    )
    merged_df: pd.DataFrame = pd.read_pickle(
        os.path.join(dataset_root, 'merged_df.pkl')
    )

    ids = set(load_df['item_id'].unique().tolist())
    products = reduce(lambda s1, s2: s1 | s2, merged_df['products'])

    with open(save_path, 'w', encoding='utf8') as fout:
        l = list(ids | products)
        l.sort()
        json.dump(l, fout)
        return l


dataset_root = 'dataset'


def clean_load_df():

    with open(
        os.path.join(dataset_root, 'available_products.json'), 'r',
        encoding='utf8'
    ) as fin:
        available_product_ids = json.load(fin)

    load_df: pd.DataFrame = pd.read_pickle(
        os.path.join(dataset_root, 'load_df_ped.pkl')
    )

    ori_n_loads = len(load_df)
    print('dropping items that are not available')
    load_df = load_df[load_df['item_id'].isin(available_product_ids)
                      ].reset_index(drop=True)
    n_dropped = ori_n_loads - len(load_df)
    print(f'{n_dropped = } ({n_dropped / ori_n_loads:.2%})')
    load_df.to_pickle(os.path.join(dataset_root, 'load_df_v2.pkl'))


def clean_merge_df():
    with open(
        os.path.join(dataset_root, 'available_products.json'), 'r',
        encoding='utf8'
    ) as fin:
        available_product_ids = set(json.load(fin))
    merged_df: pd.DataFrame = pd.read_pickle(
        os.path.join(dataset_root, 'merged_df.pkl')
    )
    print(merged_df)

    ori_n_samples = len(merged_df)
    print(
        'dropping samples that are empty after cleaning unavailable products'
    )
    merged_df.loc[:, 'products'] = merged_df['products'].map(
        lambda ps: ps & available_product_ids
    )
    merged_df = merged_df[merged_df['products'].map(len).astype(bool)
                          ].reset_index(drop=True)
    n_dropped = ori_n_samples - len(merged_df)
    print(f'{n_dropped = } ({n_dropped / ori_n_samples:.2%})')

    def clean_unavailable_products_in_loads(loads: list[dict]):
        return [
            load for load in loads if load['item_id'] in available_product_ids
        ]

    merged_df.loc[:, 'loads'] = merged_df['loads'].map(
        clean_unavailable_products_in_loads
    )
    merged_df.loc[:, 'order_id'] = merged_df['order_id'].map(int)
    print(merged_df.dtypes)
    print(len(merged_df['order_id'].unique()), len(merged_df['order_id']))
    merged_df.to_pickle(os.path.join(dataset_root, 'merged_df_v2.pkl'))


def main():
    clean_load_df()
    clean_merge_df()


main()
