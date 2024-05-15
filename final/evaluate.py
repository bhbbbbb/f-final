import numpy as np
from .data import N_RPODUCTS


def ranking_metrics(
    y_score: np.ndarray, y_true: np.ndarray, ks: list[int] | int = None
):
    """Calculate the ranking metrics, including hr, mrr, ndcg

    Args:
        y_score (np.ndarray): predicted scores in shape (#n_samples, #n_products)
        y_true (np.ndarray): ground truth in shape (#n_samples, #n_products)
        ks (list[int] | int): list of k to use. Default to [1, 5, 10, 20]

    Returns:
        dict
    """
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
