from typing import Literal
import re
import numpy as np
import pandas as pd
from .data import PID_MAPPING, AVAILABLE_PRODUCT_IDS, N_RPODUCTS


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
