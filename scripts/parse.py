import os
import re
import json
from functools import reduce, lru_cache
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import final

word_r_ = re.compile(r'[a-zA-Z-]{2,}')
tag_r_ = re.compile(r'\b[a-zA-Z- ]+\b')
# pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 400)
pd.set_option('display.min_rows', 400)
pd.set_option('display.expand_frame_repr', True)


def fuzzy_tags_clustering(
    tags_corpus: dict[int, list[str]], match_threshold: int = 90
):

    domain = final.utils.FuzzySet.from_corpus(
        tags_corpus.values(), match_threshold, verbose=True, mode='simple'
    )
    return {
        pid: [domain[tag] for tag in tags]
        for pid, tags in tags_corpus.items()
    }


def plot_metric_wrt_tags_len(
    x_range, df: pd.DataFrame, tgt_metrics: list[str], title: str
):
    for tgt_metric in tgt_metrics:
        values = df[tgt_metric]
        plt.plot(x_range, values, label=tgt_metric)
    plt.title(title)
    if len(tgt_metrics) > 1:
        plt.legend()
    plt.savefig(f'figs/{title}')
    return


def plot_metrics_wrt_k_for_splits():
    pass


def main():

    # with open('results/ckpt-13/raw_predict.json', 'r', encoding='utf8') as fin:
    from table_eval import load
    product_keywords = load('dataset/products_extracted.csv')
    RAW_PREDICT = 'results/ckpt-14/'
    # RAW_PREDICT = 'results/ckpt-14/34827a7b'
    # RAW_PREDICT = 'results/ckpt-17/4356c076'
    with open(
        os.path.join(RAW_PREDICT, 'raw_predict.json'), 'r', encoding='utf8'
    ) as fin:
        raw_predicts = json.load(fin)
        raw_predicts = {int(pid): v for pid, v in raw_predicts.items()}

    def save_enhanced_tags(
        enhanced_tags: dict[int, list[str]], name: str = 'enhanced_tags.json'
    ):
        with open(
            os.path.join(RAW_PREDICT, name), 'w', encoding='utf8'
        ) as fout:
            json.dump(enhanced_tags, fout, ensure_ascii=False, indent=4)

    @lru_cache(maxsize=2)
    def parse_raw_predict(raw_predict: str, exclude_inputs: bool) -> list[str]:
        if exclude_inputs is True:
            raw_predict = raw_predict.strip().split('\n', maxsplit=1)[-1]
        results = [tag.strip() for tag in tag_r_.findall(raw_predict.lower())]
        return results

    @lru_cache(maxsize=1)
    def predict_to_bow(raw_predict: str, exclude_inputs: bool) -> list[str]:
        if exclude_inputs is True:
            raw_predict = raw_predict.split('\n', maxsplit=1)[-1]
        return word_r_.findall(raw_predict.lower())

    PIDS = {pid: [str(pid)] for pid in final.AVAILABLE_PRODUCT_IDS}

    @lru_cache(maxsize=2)
    def _get_enhanced_tags(exclude_inputs: bool, fuzzy: bool = False):
        corpus = {
            pid: parse_raw_predict(raw, exclude_inputs=exclude_inputs)
            for pid, raw in raw_predicts.items()
        }
        if fuzzy:
            corpus = fuzzy_tags_clustering(corpus)
        return corpus

    def get_enhanced_tags(
        max_len: int = 10,
        add_pid_tag: int = 0,
        exclude_inputs: bool = False,
        fuzzy: bool = False,
    ):
        tags_corpus = {
            pid: tags[:max_len]
            for pid, tags in _get_enhanced_tags(exclude_inputs, fuzzy).items()
        }

        if add_pid_tag > 0:
            tags_corpus = {
                pid: v + ([str(pid)] * add_pid_tag)
                for pid, v in tags_corpus.items()
            }
        return tags_corpus

    def get_enhanced_words(max_len: int = None, add_pid_word: bool = True):
        words = {
            pid: predict_to_bow(raw, exclude_inputs=True)[:max_len]
            for pid, raw in raw_predicts.items()
        }
        if add_pid_word:
            words = {pid: v + [str(pid)] for pid, v in words.items()}
        return words

    # from tag_tsne import plot, similarity
    # similarity(
    #     get_enhanced_tags(exclude_inputs=True, fuzzy=True),  # tag_len=20,
    #     name='heat_enhanced'
    # )
    # return

    def cheat_query_fn(loaded_tags: list[list[str]]):
        return reduce(
            lambda a, b: a + b,
            [tags * (i + 1) for i, tags in enumerate(loaded_tags)], list()
        )

    def hybrid_score_fn(
        scores: dict[int, float], baseline_scores: dict[int, float]
    ):
        scores = scores.copy()
        for pid, s in scores.items():
            if pid not in baseline_scores:
                scores[pid] = s / 1000
        return scores

    # KS = [10, 20, 30]
    KS = [5, 10, 20]
    rows = []
    for split, merged_df in final.data.merged_df().items():
        # if split == 'train':
        #     continue
        # l9 the best
        if split == 'val':
            continue
        merged_df = final.utils.next_product_expansion(
            merged_df, target='last'
        )
        # print(merged_df)

        baseline_scores = final.utils.baseline_score(merged_df)
        metrics = final.ranking_metrics(
            baseline_scores, merged_df['y_true'], ks=KS
        )
        metrics['name'] = f'baseline/{split}'
        rows.append(metrics)

        metrics = final.ranking_metrics(
            baseline_scores.map(lambda _: {}), merged_df['y_true'], ks=KS
        )
        metrics['name'] = f'random/{split}'
        rows.append(metrics)

        def evaluate(name: str, tags: dict[int, list[str]], query_fn):
            # metrics = final.TagsEvaluator(
            #     tags,
            #     query_fn=query_fn,
            # ).cal_ranking_metrics(merged_df, ks=KS)
            tags_scores = final.TagsEvaluator(
                tags,
                query_fn=query_fn,
            ).cal_scores(merged_df)
            # tags_scores = hybrid_score_fn(tags_scores, baseline_scores)
            metrics = final.ranking_metrics(
                tags_scores, merged_df['y_true'], ks=KS
            )
            metrics['name'] = name
            rows.append(metrics)

        # evaluate(
        #     f'raw_tags_concat/{split}', final.data.product_tags_v5_en(),
        #     'concat'
        # )
        for q_name, q_fn in [
            ('sum', 'sum'), ('union', 'union'), ('last', 'last')
            # ('inter', 'intersection'),
            # ('cheat', cheat_query_fn),
        ]:
            # for q_name, q_fn in [('cheat', cheat_query_fn)]:
            # evaluate(f'pid_{q_name}/{split}', PIDS, q_fn)

            # evaluate(f'keywords_{q_name}/{split}', product_keywords, q_fn)

            # evaluate(
            #     f'raw_tags_{q_name}/{split}', final.data.product_tags_v5_en(),
            #     q_fn
            # )

            for max_len in [8, 9, 10]:
                # for max_len in range(1, 25):
                for n_pid in [0, 1, 2, 3, 4, 5, 6]:
                    enhanced_tags = get_enhanced_tags(
                        max_len=max_len,
                        add_pid_tag=n_pid,
                        fuzzy=False,
                        exclude_inputs=False,
                    )
                    if max_len == 9 and n_pid == 3:
                        pass
                        # save_enhanced_tags(
                        #     enhanced_tags, 'enhanced_tags_rev.json'
                        # )
                    evaluate(
                        f'enhanced_tags_{q_name}_l{max_len}_p{n_pid}/{split}',
                        enhanced_tags, q_fn
                    )
            # for max_len in [5, 10, 15, 20, 25, 30]:
            #     for add_n_pid in [0]:
            #         evaluate(
            #             f'enhanced_tags_{q_name}_l{max_len}_p{add_n_pid}_f/{split}',
            #             get_enhanced_tags(
            #                 max_len=max_len, add_pid_tag=add_n_pid, fuzzy=True
            #             ), q_fn
            #         )

    res_df = pd.DataFrame(rows).set_index('name')
    print(res_df)
    # res_df = res_df[res_df.index.str.contains('enhanced_tags_')
    #                 & res_df.index.str.contains('test')]
    # plot_metric_wrt_tags_len(list(range(1, 25)), res_df, ['hr@20'], 'HR@20')


if __name__ == '__main__':
    main()
