import numpy as np
import json
import pandas as pd
import final
from final import data
import final.evaluate


def baseline(df: pd.DataFrame):

    pid_mapping = {pid: i for i, pid in enumerate(final.AVAILABLE_PRODUCT_IDS)}

    def cal_scores(loads: list[dict]):
        # for load in loads:
        #     item_id = load['item_id']
        #     print(item_id)
        #     raise
        loads = [pid_mapping[load['item_id']] for load in loads]
        scores = np.random.randint(-1000, 0, len(final.AVAILABLE_PRODUCT_IDS))
        for i, pid in enumerate(loads):
            scores[pid] = i
        return scores

    def products_set_to_array(products: set[int]):
        a = np.zeros(final.N_RPODUCTS, dtype=int)
        a[[pid_mapping[pid] for pid in products]] = 1
        return a

    scores = np.array(df['loads'].map(cal_scores).to_list())
    y_true = np.array(df['products'].map(products_set_to_array).to_list())

    return final.evaluate.ranking_metrics(scores, y_true, ks=[1, 5, 10, 20])


def main():

    # dfs = data.merged_df()
    rows = []
    for split, df in data.merged_df().items():
        res = baseline(df)
        res['name'] = f'latest_first/{split}'
        rows.append(res)
    pd.DataFrame(rows).set_index('name').to_csv('results/latest_first.csv')

    # evaluate()


if __name__ == '__main__':
    main()
