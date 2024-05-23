from typing import Callable, Literal, overload
from functools import reduce
import pandas as pd
import numpy as np
from .bm25 import BM25
from ..data import N_RPODUCTS, AVAILABLE_PRODUCT_IDS, merged_df as get_merged_df


def ranking_metrics(
    y_score: np.ndarray | pd.Series,
    y_true: np.ndarray | pd.Series,
    ks: list[int] | int = None,
):
    """Calculate the ranking metrics, including hr, mrr, ndcg

    Args:
        y_score (np.ndarray): predicted scores in shape (#n_samples, #n_products)
        y_true (np.ndarray): ground truth in shape (#n_samples, #n_products)
        ks (list[int] | int): list of k to use. Default to [1, 5, 10, 20]

    Returns:
        dict
    """

    if isinstance(y_score, pd.Series):
        y_score = np.array(y_score.to_list())
    if isinstance(y_true, pd.Series):
        y_true = np.array(y_true.to_list())

    assert y_score.shape[
        -1] == N_RPODUCTS, f'{y_score.shape[-1] = } which != {N_RPODUCTS}'
    assert y_true.shape[
        -1] == N_RPODUCTS, f'{y_true.shape[-1] = } which != {N_RPODUCTS}'
    assert len(y_true) == len(y_score)
    if ks is None:
        ks = [1, 5, 10, 20]
    elif isinstance(ks, int):
        ks = [ks]

    y_rank = np.argsort(-y_score, axis=1)
    mrr_weights = 1 / (np.arange(max(ks)) + 1)
    dcg_weights = 1 / np.log2(2 + np.arange(max(ks)))
    idcg_cumulative_weights = dcg_weights.cumsum()

    def _cal_metrics():
        for k in ks:
            y_rank_at_k = y_rank[:, :k]
            y_rank_true_at_k = np.take_along_axis(y_true, y_rank_at_k, axis=1)
            n_true = y_true.sum(axis=1).clip(max=k)
            dcg_ = (dcg_weights[:k] * y_rank_true_at_k).sum(axis=1)
            idcg_ = idcg_cumulative_weights[n_true - 1]

            ndcg = (dcg_ / idcg_).mean()
            mrr = (mrr_weights[:k] * y_rank_true_at_k).sum(axis=1).mean()
            hr = (y_rank_true_at_k.sum(axis=1) / n_true).mean()
            yield f'hr@{k}', hr
            yield f'mrr@{k}', mrr
            yield f'ndcg@{k}', ndcg

    order = [f'{met}@{k}' for met in ['hr', 'mrr', 'ndcg'] for k in ks]
    res = dict(_cal_metrics())
    return {key: res[key] for key in order}


def iou_score(query: list[str], documents: list[list[str]]):

    if not isinstance(query, set):
        query = set(query)

    def scores():
        for i, document in enumerate(documents):
            if not isinstance(document, set):
                document = set(document)
            yield i, len(document & query) / len(document | query)

    # return list(iou_scores())
    return {i: score for i, score in scores() if score > 0}


class TagsEvaluator:

    def __init__(
        self,
        product_tags: dict[int, list[str]],
        query_reduce: Literal['union', 'concat'] = 'concat',
        score_algorithm: Literal['bm25', 'iou'] = 'bm25',
    ):

        assert set(product_tags) >= set(AVAILABLE_PRODUCT_IDS)
        if len(product_tags) > len(AVAILABLE_PRODUCT_IDS):
            product_tags = {
                pid: product_tags[pid]
                for pid in AVAILABLE_PRODUCT_IDS
            }

        self.product_tags = product_tags
        self.corpus = list(product_tags.values())
        if score_algorithm == 'bm25':
            bm25 = BM25(self.corpus)
            self.scores_fn = bm25.get_scores
        else:
            assert query_reduce == 'union', 'when score using IoU, qurey_reduce only accept "union"'
            self.scores_fn = iou_score

        self.query_reduce_fn = (
            (lambda a, b: a + b) if query_reduce == 'concat' else
            (lambda a, b: set(a) | set(b))
        )

    def cal_scores(self, merged_df: pd.DataFrame):
        """Calculate recommendation scores of the orders in given merged_df

        Args:
            merged_df (pd.DataFrame): merged_df. Can be obtained via `final.data.merged_df()[split]`
        """

        def cal_score(loads: list[dict]):
            query = reduce(
                self.query_reduce_fn,
                [self.product_tags[pid] for pid in loads],
                list(),
            )
            scores_dict = self.scores_fn(query, self.corpus)
            scores = np.random.rand(len(AVAILABLE_PRODUCT_IDS)) - 11.
            for pid, score in scores_dict.items():
                scores[pid] = score
            return scores

        return merged_df['loaded_pids'].map(cal_score)

    @overload
    def cal_ranking_metrics(
        self,
        *,
        ks: list[int] = ...,
    ) -> dict[Literal['train', 'val', 'test'], dict[float]]:
        ...

    @overload
    def cal_ranking_metrics(
        self,
        merged_df: pd.DataFrame,
        *,
        ks: list[int] = ...,
    ) -> dict[float]:
        ...

    def cal_ranking_metrics(
        self,
        merged_df: pd.DataFrame | None = None,
        *,
        ks: list[int] = [1, 5, 10, 20],
    ):

        if merged_df is not None:
            return ranking_metrics(
                self.cal_scores(merged_df), merged_df['y_true'], ks=ks
            )

        return {
            split: self.cal_ranking_metrics(df, ks=ks)
            for split, df in get_merged_df().items()
        }
