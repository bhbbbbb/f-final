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
    load_df = load_df[domain.str.match(r'(mob\.)?i3fresh.tw')]
    n_drop_samples = len(_load_df) - len(load_df)
    print('dropping samples those are not in i3.fresh')
    print(
        f'{n_drop_samples} samples ({n_drop_samples / n_samples_orig:.2%}) have been dropped.'
    )

    path = load_df.url_now.map(lambda url: urlparse(url).path)

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
    return load_df

    # remaining_path = path[~reduce(lambda a, b: a | b, filter_path())]
    # assert remaining_path.empty
    # # print(path.value_counts())
    # print(remaining_path.value_counts())


def is_sorted(s: pd.Series):
    if len(s) == 1:
        return True

    s = s.to_numpy()
    return ((s[1:] - s[:-1]) < 0).sum() == 0


def mp_sorting_task(session_id):
    global load_df
    session_df: LoadDFSchema = load_df[load_df.session_id == session_id]

    if len(vc := session_df['uuid_ind'].value_counts()) > 1:
        print('multiple_user_detected', session_id, vc)
        with open(
            f'muluser_sessions/{session_id}', 'w', encoding='utf8'
        ) as fout:
            fout.write(str(vc))

    if is_sorted(session_df['timestamp']):
        return session_df.reset_index(drop=True)

    print(session_id, 'is now sorted.')
    return session_df.sort_values('timestamp').reset_index(drop=True)


def mp_init(df):
    global load_df
    load_df = df


# Sort every sessions
def sort_df_by_timestamp(load_df: LoadDFSchema) -> LoadDFSchema:

    return load_df.sort_values(by=['uuid_ind', 'session_id', 'timestamp']
                               ).reset_index(drop=True)
    # with Pool(processes=4, initializer=mp_init, initargs=(load_df, )) as pool:

    #     uniques = load_df[['session_id', 'uuid_ind']].drop_duplicates()
    #     res = pool.imap(mp_sorting_task, uniques, chunksize=128)

    #     sorted_load_df = pd.concat(tqdm(res, total=len(uniques)), axis=0)
    # assert len(sorted_load_df) == len(load_df)
    # return sorted_load_df.reset_index(drop=True)


def main():

    # load_df: LoadDFSchema = pd.read_pickle(
    #     os.path.join(DATASET_ROOT, 'load_df.pkl')
    # )
    # load_df = cleaning(load_df)
    # load_df.to_pickle(os.path.join(DATASET_ROOT, 'load_df_ped_.pkl'))

    # return
    load_df: LoadDFSchema = pd.read_pickle(
        os.path.join(DATASET_ROOT, 'load_df_ped_.pkl')
    )

    # for session_id in tqdm(load_df['session_id'].unique()):
    #     session_df = load_df[load_df['session_id'] == session_id]
    #     if len(vc := session_df['uuid_ind'].value_counts()) > 1:
    #         print(session_id, vc)
    #         raise
    load_df = load_df.astype({'session_id': 'int64'})
    print('start sorting')
    # load_df = sort_df_by_timestamp(load_df)
    # TODO: Drop continuous duplicates

    # load_df.to_pickle(os.path.join(DATASET_ROOT, 'load_df_ped.pkl'))
    return


if __name__ == '__main__':
    main()
