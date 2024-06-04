from functools import reduce
import re
from urllib.parse import urlparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from pprint import pprint

# pd.set_option('display.width', 10000)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 100)
# pd.set_option('max_colwidth', None)
# pd.set_option('display.expand_frame_repr', False)


class LoadDFSchema(pd.DataFrame):
    # web_id: pd.Series
    timestamp: pd.Series
    session_id: pd.Series
    url_referrer: pd.Series
    url_now: pd.Series
    device_type: pd.Series
    landing_count: pd.Series
    uuid_ind: pd.Series


def print_bincount_series(bc: pd.Series, sep: str = ','):
    for idx in bc.index:
        print(idx, end=sep)
    print()
    for val in bc.values:
        print(val, end=sep)
    print()
    return


def plot_unique_session_length_dist(
    load_df: pd.DataFrame, outputfig: str = 'figs/unique_session_len_dist.png'
):
    unique_session_length = load_df[['uuid_ind', 'session_id', 'item_id'
                                     ]].groupby(['uuid_ind', 'session_id']
                                                )['item_id'].nunique()
    bc = unique_session_length.value_counts()
    plt.bar(bc.index, bc.values, log=True)
    plt.title('unique session length distribution')
    # plt.show()
    plt.savefig(outputfig)
    plt.close()
    print('unique session length:')
    print_bincount_series(bc)
    print('average: ', unique_session_length.mean())


def plot_session_length_dist(
    load_df: LoadDFSchema, outputfig: str = 'figs/session_len_dist.png'
):
    session_length: pd.Series[int] = load_df.groupby(
        ['uuid_ind', 'session_id']
    ).size()

    bc = session_length.value_counts()
    plt.bar(bc.index, bc.values, log=True)
    plt.title('session length distribution')
    # plt.show()
    plt.savefig(outputfig)
    plt.close()
    print('session length: ')
    print_bincount_series(bc)
    print('average', session_length.mean())


def plot_num_product_per_order_dist(
    purchase_df: pd.DataFrame,
    outfig: str = 'figs/n_product_per_order_dist.png'
):

    purchase_length = purchase_df['products'].map(len)
    bc = purchase_length.value_counts()
    plt.bar(bc.index, bc.values, log=True)
    plt.title('#product per order distribution')
    # plt.show()
    plt.savefig(outfig)
    plt.close()
    print('# of products per order')
    print_bincount_series(bc)
    print('average', purchase_length.mean())
    return


def eda(load_df: LoadDFSchema):

    counts = load_df.groupby(['uuid_ind', 'session_id']).size()
    print('# of all sessions,', len(counts))
    # print(f'avg loads per session = {counts.mean()}')
    plot_session_length_dist(
        load_df, outputfig='figs/available_session_len_dist.png'
    )
    plot_unique_session_length_dist(
        load_df, outputfig='figs/available_unique_session_len_dist.png'
    )
    # plot_num_product_per_order_dist(purchase_df)
    return load_df


def check_if_all_purchased_products_in_loads(merged_df: pd.DataFrame):

    def _check():
        loads = merged_df['loads'].map(lambda ds: {d['item_id'] for d in ds})
        for order_id, products, loads in zip(
            merged_df['order_id'], merged_df['products'], loads
        ):
            # print(products)
            # print(loads)
            # input('..')
            if not (products <= loads):
                n = len(products - loads)
                # print(f'In order:{order_id}, {n} products is not in loads')
                yield n
            else:
                yield 0

    n_not_in = pd.Series(list(_check()))
    print('# of purchased items not in corresponding browsing sequences')
    bc = n_not_in.value_counts()
    print_bincount_series(bc)
    print('avg, ', n_not_in.mean())
    plt.bar(bc.index, bc.values)
    plt.title('# of purchased items not in corresponding browsing sequences')
    # plt.show()
    plt.savefig('figs/n_purchased_not_in_loads.png')
    plt.close()


def baseline(merged_df: pd.DataFrame):
    pass


def main():

    # load_df: pd.DataFrame = pd.read_pickle('./dataset/load_df_ped.pkl')
    # purchase_df: pd.DataFrame = pd.read_pickle('./dataset/purchase_df_ped.pkl')
    # merged_df = pd.DataFrame = pd.read_pickle('./dataset/merged_df.pkl')
    # merged_df = pd.DataFrame = pd.read_pickle('./dataset/merged_df.pkl')
    import final
    load_df = final.data.load_df()

    # plot_num_product_per_order_dist(merged_df)
    # check_if_all_purchased_products_in_loads(merged_df)
    eda(load_df)
    return
    # load_df = load_df[:10000]
    load_sessions = load_df[[
        'uuid_ind', 'session_id'
    ]].apply(lambda s: (s.iloc[0], s.iloc[1]), axis=1)
    valid_sessions = set(
        merged_df[['uuid_ind', 'session_id'
                   ]].apply(lambda s: (s.iloc[0], s.iloc[1]), axis=1).values
    )
    # print(t)
    # print(valid_sessions)
    load_valid_df = load_df[load_sessions.isin(valid_sessions)].reset_index(
        drop=True
    )

    print('-' * 100)
    eda(load_valid_df)
    # print(
    #     'num of valid sessions (sessions with order placed) = ',
    #     len(load_valid_df)
    # )
    # plot_session_length_dist(load_valid_df, 'figs/valid_session_len_dist.png')
    # plot_unique_session_length_dist(
    #     load_valid_df, 'figs/valid_unique_session_len_dist.png'
    # )
    # # check_if_all_purchased_products_in_loads(merged_df)


main()
