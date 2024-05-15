import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_n_order_per_session(
    purchase_df: pd.DataFrame,
    outfig: str = 'figs/n_order_per_session_dist.png'
):

    bc = purchase_df['products'].map(len).value_counts()
    plt.bar(bc.index, bc.values, log=True)
    plt.title('#product per order distribution')
    # plt.show()
    plt.savefig(outfig)
    plt.close()
    return


def main():
    purchase_df: pd.DataFrame = pd.read_pickle('./dataset/purchase_df.pkl')
    assert (purchase_df['web_id'] == 'i3fresh').sum() == len(purchase_df)
    assert (purchase_df['event_type'] == 'purchase').sum() == len(purchase_df)

    purchase_df = purchase_df.drop(['web_id', 'event_type'], axis=1)
    purchase_df = purchase_df.astype({'session_id': 'int64'})

    order_ids = purchase_df['purchasing_detail'].map(lambda d: d['order_id'])
    # total_prices = purchase_df['purchasing_detail'].map(lambda d: d['total_price'])

    purchase_df = pd.concat(
        [purchase_df, order_ids.rename('order_id')], axis=1
    )

    print('dropping duplicated orders...')
    purchase_df_ = purchase_df
    purchase_df = purchase_df.drop_duplicates(subset=['order_id']
                                              ).reset_index(drop=True)

    print(purchase_df)
    n_dropped = len(purchase_df_) - len(purchase_df)
    print(f'{n_dropped} rows have been dropped.')

    purchased_products = purchase_df['purchasing_detail'].map(
        lambda d: {int(o['product_id'])
                   for o in d['details']}
    )
    purchase_df = pd.concat(
        [purchase_df, purchased_products.rename('products')], axis=1
    )

    # purchase_df['products'].map(check_products_in_order_duplicated)

    print('dropping orders those contain no product')
    pd.set_option('display.width', 10000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    purchase_df = purchase_df[purchase_df['products'].map(len) == 0]
    purchase_df.head().to_csv('figs/purchase_df_no_product_order_exp.csv')
    print(purchase_df.head())
    raise
    purchase_df_ = purchase_df
    purchase_df = purchase_df[purchase_df['products'].map(len) > 0]
    n_dropped = len(purchase_df_) - len(purchase_df)
    print(f'{n_dropped} rows have been dropped.')

    purchase_df.to_pickle('./dataset/purchase_df_ped.pkl')


main()
