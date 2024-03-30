from multiprocessing import Pool
import os
from functools import reduce
import re
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm

DATASET_ROOT = './dataset'


class LoadDFSchema(pd.DataFrame):
    # web_id: pd.Series
    timestamp: pd.Series
    session_id: pd.Series
    url_referrer: pd.Series
    url_now: pd.Series
    device_type: pd.Series
    landing_count: pd.Series
    uuid_ind: pd.Series


def cleaning(load_df: LoadDFSchema):
    assert (load_df['event_type'] == 'load').sum() == len(load_df)
    assert (load_df['web_id'] == 'i3fresh').sum() == len(load_df)
    load_df = load_df.drop(['event_type', 'web_id'], axis=1)
    load_df = load_df.astype({'session_id': 'int64'})

    n_samples_orig = len(load_df)

    # # Drop
    # purchase_df: pd.DataFrame = pd.read_pickle('./dataset/purchase_df_ped.pkl')
    # valid_users = set(purchase_df['uuid_ind'].unique()
    #                   ) & set(load_df['uuid_ind'].unique())
    # print(
    #     f'load_user_count={len(load_df["uuid_ind"].unique())}, purchase_user_count={len(purchase_df["uuid_ind"].unique())}, valid_count={len(valid_users)}'
    # )
    # print('dropping invalid users (users not in purchase_df)')
    # load_df_ = load_df
    # load_df = load_df[load_df['uuid_ind'].isin(valid_users)].reset_index(
    #     drop=True
    # )
    # n_drop_samples = len(load_df_) - len(load_df)
    # print(
    #     f'{n_drop_samples} samples ({n_drop_samples / n_samples_orig:.2%}) have been dropped.'
    # )

    # Drop self-referrered rows
    load_df_ = load_df
    load_df: LoadDFSchema = load_df[load_df.url_now != load_df.url_referrer
                                    ].reset_index(drop=True)
    n_drop_samples = len(load_df_) - len(load_df)
    print('dropping samples those are self-refferred')
    print(
        f'{n_drop_samples} samples ({n_drop_samples / n_samples_orig:.2%}) have been dropped.'
    )

    # Find all paths of urls
    domain = load_df.url_now.map(lambda url: urlparse(url).netloc)

    _load_df = load_df
    load_df = load_df[domain.str.match(r'(mob\.)?i3fresh.tw')].reset_index(
        drop=True
    )
    n_drop_samples = len(_load_df) - len(load_df)
    print('dropping samples those are not in i3.fresh')
    print(
        f'{n_drop_samples} samples ({n_drop_samples / n_samples_orig:.2%}) have been dropped.'
    )

    path = load_df.url_now.map(lambda url: urlparse(url).path
                               ).rename('url_path')
    load_df = pd.concat([load_df, path.to_frame()], axis=1)

    path_regex_to_drop = {
        'member': r'/+member\.(?:php|html)',
        'newslist': r'/+newslist\.(?:php|html)',
        'specialty': r'/+specialty\.(?:php|html)',
        'authenticate': r'/+authenticate\.(?:php|html)',
        'contact': r'/+contact(_proposal)?\.(?:php|html)',
        'forget': r'/+forget\.(?:php|html)',
        'inquiry': r'/+inquiry\.(?:php|html)',
        'sentok': r'/+sentok\.(?:php|html)',
        'brand': r'/+brand\.(?:php|html)',
        'precursor': r'/+precursor\.(?:php|html)',
        'new_ecpay_payment': r'/+new_ecpay_payment(_3d)?\.php',
        'shopping_cart': r'/+shoppingcart\.(?:html|php)',
        'home': r'/+((?:index|inpage|favorable|cheap)\.(?:html|php))?$',
        'insearch': r'/+insearch\.html',
        'modify': r'/+modify\.html',
        'order': r'/+(?:mob|new_|newmob)?order(?:ok|_\d+)?\.html',
        'orderview': r'/+(?:orderview|oview)(_[\w\d_]+)?\.html',
        'login': r'/+login(_\d+)?\.(?:php|html)',
        'register': r'/+register(?:ed|start)?\.(?:php|html)',
        'getbouns': r'/+(?:get|set)?bonus(_\S+)?\.html',
        'getgift': r'/+getgift_\S+?\.html',
        'fag': r'/+faq(_\d+)?\.(?:html|php)',
        'bulletin': r'/+bulletin_(\d+)\.html',
        'cheapview': r'/+cheapview_(\d+)\.(?:php|html)',
        'bastchoice': r'/+bastchoice\.(?:php|html)',
        'inpage': r'/+(?:inpage|index)_(\d+)(_(\d+))?\.html',
    }

    def re_match(regexes: dict, path_ser: pd.Series) -> pd.Series:

        def gen():
            for name, pattern in regexes.items():
                matches = path_ser.str.match(pattern)
                # print(name, matches.sum())
                yield matches

        return reduce(lambda a, b: a | b, gen())

    _load_df = load_df
    print(
        'dropping samples those are useless browsing histories (not an item)'
    )
    to_drop = ~re_match(path_regex_to_drop, path)  # pylint: disable=invalid-unary-operand-type
    load_df = load_df[to_drop].reset_index(drop=True)
    path = path[to_drop].reset_index(drop=True)
    n_drop_samples = len(_load_df) - len(load_df)
    print(
        f'{n_drop_samples} samples ({n_drop_samples / n_samples_orig:.2%}) have been dropped.'
    )

    item_regexes = {
        'cheap': r'/+cheap_(\d+)\.html',
        'favorable': r'/+favorable_(\d+)\.html',
    }

    assert re_match(item_regexes, path).sum() == len(load_df), str(
        path[~re_match(item_regexes, path)].value_counts()
    )

    item_ids = path.map(
        lambda p: re.match(r'/+(?:cheap|favorable)_(\d+)\.html', p).group(1)
    ).rename('item_id')
    item_types = path.str.match(item_regexes['cheap']
                                ).map({
                                    True: 'cheap',
                                    False: 'favorable'
                                }).rename('item_type')

    def find_referrer_cat(url: str):
        match = re.search(r'/+(?:inpage|index)_(\d+)(_(\d+))?\.html', url)
        if match is None:
            return pd.Series([None, None], index=['main_cat', 'sub_cat'])
        return pd.Series(
            [match.group(1), match.group(3)], index=['main_cat', 'sub_cat']
        )

    item_cats = load_df.url_referrer.apply(find_referrer_cat)

    load_df = pd.concat([load_df, item_ids, item_types, item_cats], axis=1)

    print("Sorting load_df by ['uuid_ind', 'session_id', 'timestamp']")
    load_df = load_df.sort_values(by=['uuid_ind', 'session_id', 'timestamp']
                                  ).reset_index(drop=True)

    print('Dropping continuous duplicates')

    def mark_continuous_duplicates():
        _df = load_df[['uuid_ind', 'session_id']]
        _url_paths = load_df['url_path']
        yield True
        for i, prev_path, cur_path in zip(
            range(1, len(_df)), _url_paths, _url_paths[1:]
        ):
            if prev_path == cur_path:
                prev_row = _df.iloc[i - 1]
                cur_row = _df.iloc[i]
                if (prev_row == cur_row).all():
                    yield False
                    continue
            yield True

    load_df_ = load_df
    res = pd.Series(
        list(tqdm(mark_continuous_duplicates(), total=len(load_df)))
    )
    print(load_df[:50][['uuid_ind', 'session_id', 'url_path', 'url_now']])
    load_df = load_df[res]  #.reset_index(drop=True)
    print(load_df[:50][['uuid_ind', 'session_id', 'url_path', 'url_now']])
    n_drop_samples = len(load_df_) - len(load_df)
    load_df = load_df.reset_index(drop=True)

    print(f'{n_drop_samples} samples have been dropped.')
    return load_df


def main():

    load_df: LoadDFSchema = pd.read_pickle(
        os.path.join(DATASET_ROOT, 'load_df.pkl')
    )
    load_df = cleaning(load_df)
    load_df.to_pickle(os.path.join(DATASET_ROOT, 'load_df_ped_.pkl'))
    return

    # load_df: LoadDFSchema = pd.read_pickle(
    #     os.path.join(DATASET_ROOT, 'load_df_ped_.pkl')
    # )

    # load_df.to_pickle(os.path.join(DATASET_ROOT, 'load_df_ped.pkl'))
    # return


if __name__ == '__main__':
    main()
