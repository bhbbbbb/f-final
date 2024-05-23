import re
import numpy as np
import pandas as pd
from .data import PID_MAPPING, AVAILABLE_PRODUCT_IDS


def url_to_pid(url: str):
    match = re.search(r'(?:favorable|cheap)_(\d+)', url)
    return int(match.group(1))


def chinese_tags_to_bow(
    tags: list[str] | dict[int, list[str]] | list[list[str]]
) -> list[str]:

    def _gen(_tags: list[str]):
        for tag in _tags:
            for character in tag:
                yield character

    if not isinstance(tags, dict) and isinstance(tags[0], str):
        return list(_gen(tags))

    if isinstance(tags, dict):
        return {pid: list(_gen(ts)) for pid, ts in tags.items()}

    return [list(_gen(ts)) for ts in tags]


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
