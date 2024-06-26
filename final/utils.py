from typing import Literal
import re
import numpy as np
import pandas as pd
from .data import PID_MAPPING, AVAILABLE_PRODUCT_IDS, N_RPODUCTS
from thefuzz import fuzz


def url_to_pid(url: str):
    match = re.search(r'(?:favorable|cheap)_(\d+)', url)
    return int(match.group(1))


def _tags_to_bow(
    tags: list[str] | dict[int, list[str]] | list[list[str]],
    transform_fn,
) -> list[str]:

    if not isinstance(tags, dict) and isinstance(tags[0], str):
        return list(transform_fn(tags))

    if isinstance(tags, dict):
        return {pid: list(transform_fn(ts)) for pid, ts in tags.items()}

    return [list(transform_fn(ts)) for ts in tags]


def english_tags_to_bow(
    tags: list[str] | dict[int, list[str]] | list[list[str]]
) -> list[str]:
    """Tags to bag-of-words

    E.g. 
    >>> english_tags_to_bow(['hello world', 'I am a tag'])
    >>> ['hello', 'world', 'I', 'am', 'a', 'tag']

    Args:
        tags (list[str] | dict[int, list[str]] | list[list[str]]): list of tags

    Returns:
        Bag of words
    """

    def _gen(_tags: list[str]):
        for tag in _tags:
            yield from re.split(r'\s+', tag)

    return _tags_to_bow(tags, _gen)


def chinese_tags_to_bow(
    tags: list[str] | dict[int, list[str]] | list[list[str]]
) -> list[str]:

    def _gen(_tags: list[str]):
        for tag in _tags:
            for character in tag:
                yield character

    return _tags_to_bow(tags, _gen)


def baseline_score(merged_df: pd.DataFrame):
    """Score products using their indices in load sequences.

    E.g. Given loads(list of pids): [3, 10, 5, 3], then score(p3)==3, score(p10)==2, score(p5)==1.
        The rest products would have random negative scores.

    Args:
        loads (list[dict]): list of product ids
    """

    def cal_score(loaded_pids: list[int]):
        loads = [PID_MAPPING[pid] for pid in loaded_pids]
        return {pid: (i + 1) for i, pid in enumerate(loads)}

    return merged_df['loaded_pids'].map(cal_score)


def next_product_expansion(
    merged_df: pd.DataFrame,
    target: Literal['last', 'iterative'] = 'last',
):
    merged_df = merged_df[merged_df['loaded_pids'].map(len) >= 2]
    if target == 'last':

        def drop_last(seq: list[int]):
            return seq[:-1]

        def get_last(seq: list[int]):
            return PID_MAPPING[seq[-1]]

        loaded_pids = merged_df['loaded_pids'].map(drop_last)
        y_true_id = merged_df['loaded_pids'].map(get_last)
        y_true = np.zeros((len(y_true_id), N_RPODUCTS), dtype=int)
        np.put_along_axis(
            y_true,
            y_true_id.to_numpy().reshape(-1, 1), 1, axis=1
        )

        return pd.DataFrame(
            {
                'uuid_ind': merged_df['uuid_ind'],
                'loaded_pids': loaded_pids,
                'y_true': list(y_true),
                'y_true_id': y_true_id.map(lambda i: AVAILABLE_PRODUCT_IDS[i])
            }
        )
    assert target == 'iterative'

    def explode():
        for pids in merged_df['loaded_pids']:
            for seq_len in range(1, len(pids)):
                seq = pids[:seq_len]
                y = pids[seq_len]
                y_true = np.zeros(N_RPODUCTS, dtype=int)
                y_true[PID_MAPPING[y]] = 1
                yield {'loaded_pids': seq, 'y_true': y_true, 'y_true_id': y}

    return pd.DataFrame(explode())


class FuzzySet:

    def __init__(
        self, threshold: int, mode: Literal['simple', 'sort', 'set'] = 'sort'
    ):
        self.threshold = threshold
        self.domain: set[str] = set()
        MAPPER = {
            'simple': fuzz.ratio,
            'sort': fuzz.token_sort_ratio,
            'set': fuzz.token_set_ratio,
        }
        self.fuzz_fn = MAPPER[mode]
        return

    def get(self, item: str, default=None):
        try:
            return self.__getitem__(item)
        except KeyError:
            return default

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, item: str):
        for tag in self.domain:
            if self.fuzz_fn(item, tag) >= self.threshold:
                return tag
        raise KeyError(f'key: {item} not found.')

    def add(self, item: str, verbose: bool = False):
        match = self.get(item, None)
        if match is None:
            self.domain.add(item)

        if verbose:
            if match is None:
                # print(f'item: "{item}" not found. added.')
                pass
            elif match != item:
                print(f'item: "{item}" matches with "{match}". skipped.')
        return

    @classmethod
    def from_corpus(
        cls,
        corpus: list[list[str]],
        threshold: int,
        mode: Literal['simple', 'sort', 'set'] = 'sort',
        verbose: bool = False,
    ):
        corpus = np.array(sum(corpus, start=list()))
        perm = np.random.permutation(len(corpus))
        domain = cls(threshold, mode=mode)
        for tag in corpus[perm]:
            domain.add(tag)
        print(f'{len(corpus) = }, {len(domain) = }')
        return domain
