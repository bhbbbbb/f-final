import os
import json
import numpy as np
from itertools import product
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import seaborn as sns
import final


def tags_to_np(tags_corpus: dict[int, list[str]], tag_len: int = None):
    tags_corpus = {pid: tags[:tag_len] for pid, tags in tags_corpus.items()}
    corpus = set(sum(tags_corpus.values(), list()))

    def one_hot(idx: int):
        a = np.zeros(len(corpus), dtype=int)
        a[idx] = 1
        return a

    corpus = {tag: one_hot(idx) for idx, tag in enumerate(corpus)}

    tags = np.array(
        [
            np.array([corpus[tag] for tag in tags_corpus[pid]]).sum(axis=0)
            for pid in final.AVAILABLE_PRODUCT_IDS
        ]
    )
    return tags


def plot_tsne(tags_corpus: dict[int, list[str]], tag_len: int, name: str):

    tags = tags_to_np(tags_corpus, tag_len)
    print(tags)
    print(tags.shape)
    # TruncatedSVD(n_components=)
    tsne = TSNE(n_components=2, perplexity=5)
    tags = tsne.fit_transform(tags)
    plt.scatter(tags[:, 0], tags[:, 1])
    plt.savefig(f'figs/{name}')
    plt.close()


def similarity(
    tags_corpus: dict[int, list[str]],
    name: str,
    products_to_show: list[int],
):
    tags = tags_to_np(tags_corpus, None)
    # products_to_show = np.random.permutation(len(tags))[:n_products]
    tags = tags[products_to_show]
    print(tags)
    print(tags.shape)
    similarity = cosine_similarity(tags, tags)
    sns.heatmap(similarity)
    os.makedirs('figs/heat', exist_ok=True)
    plt.savefig(f'figs/heat/{name}')
    plt.close()


def fuzz_similarity(
    tags_corpus: dict[int, list[str]],
    name: str,
    products_to_show: list[int],
):
    from thefuzz import fuzz
    tags = np.array(
        [' '.join(tags_corpus[pid]) for pid in final.AVAILABLE_PRODUCT_IDS]
    )
    tags = tags[products_to_show]
    N = len(products_to_show)
    ratios = np.zeros((N, N))
    for i, j in product(range(N), range(N)):
        if i >= j:
            continue
        ratios[i, j] = fuzz.token_sort_ratio(tags[i], tags[j]) / 100.

    ratios = ratios + ratios.T
    ratios[np.diag_indices(N)] = 1.
    sns.heatmap(ratios)
    os.makedirs('figs/fuzz_heat', exist_ok=True)
    plt.savefig(f'figs/fuzz_heat/{name}')
    plt.close()


def get_enhanced_tags():
    RAW_PREDICT = 'results/ckpt-14/'
    with open(
        os.path.join(RAW_PREDICT, 'enhanced_tags_rev.json'), 'r',
        encoding='utf8'
    ) as fin:
        enhanced_tags = json.load(fin)
        return {int(pid): v for pid, v in enhanced_tags.items()}


def main():
    from table_eval import load
    product_keywords = load('dataset/products_extracted.csv')
    enhanced_tags = get_enhanced_tags()

    # plot_tsne(enhanced_tags, 5, 'tsne_enhanced_tags')
    # plot_tsne(final.data.product_tags_v5_en(), 5, 'tsne_raw_v5_en')

    products_to_show = np.random.permutation(final.N_RPODUCTS)[:50]
    print(products_to_show)
    # fuzz_similarity(product_keywords, 'heat_keywords', products_to_show)
    # fuzz_similarity(enhanced_tags, 'heat_enhanced_tags', products_to_show)
    # fuzz_similarity(
    #     final.data.product_tags_v5_en(), 'heat_raw_v5_en', products_to_show
    # )
    similarity(product_keywords, 'heat_keywords', products_to_show)
    similarity(enhanced_tags, 'heat_enhanced_tags', products_to_show)
    similarity(
        final.data.product_tags_v5_en(), 'heat_raw_v5_en', products_to_show
    )
    pass


if __name__ == '__main__':
    main()
