

## Install and Setup


```sh
pip install git+https://github.com/bhbbbbb/f-final
```

## Basic

- Easiest way to evaluate tags for all products.

```python
>>> final.TagsEvaluator(product_tags).cal_ranking_metrics(final.data.merged_df()['train'])
{
    'hr@1': ...,
    'hr@5': ...,
    'hr@10': ...,
    ...
}
```

Check [general_demo](./demos/demo.ipynb) for more demonstration of basic usage.

## `final.data`

load the cleaned and preprocessed data automatically (data would be downloaded and cached in directory `dataset`).

check [demo](./demos/data.ipynb) for more details.



## Results

- `baseline`: score of a product is its highest index in the load sequence.
- `keyword`: keywords directly from i3fresh
- `raw_tags_v4`: raw tags from `Huan/ver4.csv`
- `*_bow`: tags/keywords preprocessed using bag-of-words.


| name                  |   hr@1 |   hr@5 |   hr@10 |   hr@20 |   mrr@1 |   mrr@5 |   mrr@10 |   mrr@20 |   ndcg@1 |   ndcg@5 |   ndcg@10 |   ndcg@20 |
|:----------------------|:-------|:-------|:--------|:--------|:--------|:--------|:---------|:---------|:---------|:---------|:----------|:----------|
| baseline/train        | 0.8227 | 0.8653 |  0.893  |  0.8999 |  0.8227 |  1.0662 |   1.0887 |   1.0915 |   0.8227 |   0.8354 |    0.849  |    0.852  |
| keywords/train        | 0.5435 | 0.6666 |  0.7379 |  0.7767 |  0.5435 |  0.7596 |   0.7927 |   0.8029 |   0.5435 |   0.5983 |    0.6295 |    0.644  |
| keywords_bow/train    | 0.4328 | 0.5139 |  0.5641 |  0.6036 |  0.4328 |  0.5846 |   0.606  |   0.6151 |   0.4328 |   0.4596 |    0.4804 |    0.4945 |
| raw_tags_v4_q/train   | 0.7857 | 0.8391 |  0.879  |  0.8986 |  0.7857 |  1.0202 |   1.0445 |   1.0504 |   0.7857 |   0.8043 |    0.8224 |    0.8301 |
| raw_tags_v4/train     | 0.7573 | 0.8488 |  0.8872 |  0.9009 |  0.7573 |  1.0115 |   1.0364 |   1.0409 |   0.7573 |   0.8    |    0.8178 |    0.8233 |
| raw_tags_v4_bow/train | 0.6121 | 0.6094 |  0.6626 |  0.7012 |  0.6121 |  0.7569 |   0.7803 |   0.7896 |   0.6121 |   0.5884 |    0.6105 |    0.6244 |
| baseline/val          | 0.8388 | 0.8643 |  0.8964 |  0.9039 |  0.8388 |  1.0942 |   1.1227 |   1.1258 |   0.8388 |   0.8392 |    0.8553 |    0.8586 |
| keywords/val          | 0.5806 | 0.6821 |  0.7528 |  0.7962 |  0.5806 |  0.8089 |   0.8446 |   0.8569 |   0.5806 |   0.6229 |    0.6539 |    0.6706 |
| keywords_bow/val      | 0.4819 | 0.5368 |  0.5872 |  0.6252 |  0.4819 |  0.6397 |   0.6636 |   0.6735 |   0.4819 |   0.4915 |    0.5127 |    0.5268 |
| raw_tags_v4_q/val     | 0.811  | 0.8375 |  0.8816 |  0.904  |  0.811  |  1.0532 |   1.0825 |   1.0896 |   0.811  |   0.8121 |    0.8322 |    0.8412 |
| raw_tags_v4/val       | 0.7813 | 0.8514 |  0.8917 |  0.9076 |  0.7813 |  1.0479 |   1.0771 |   1.0827 |   0.7813 |   0.8115 |    0.8305 |    0.837  |
| raw_tags_v4_bow/val   | 0.6553 | 0.6274 |  0.6792 |  0.7205 |  0.6553 |  0.8036 |   0.83   |   0.8409 |   0.6553 |   0.6151 |    0.6373 |    0.6527 |
| baseline/test         | 0.8454 | 0.8743 |  0.9034 |  0.9106 |  0.8454 |  1.0959 |   1.1197 |   1.1223 |   0.8454 |   0.8493 |    0.8636 |    0.8666 |
| keywords/test         | 0.5559 | 0.6819 |  0.7567 |  0.7941 |  0.5559 |  0.794  |   0.8281 |   0.838  |   0.5559 |   0.614  |    0.6464 |    0.6605 |
| keywords_bow/test     | 0.4378 | 0.5063 |  0.5539 |  0.6019 |  0.4378 |  0.5999 |   0.6208 |   0.6311 |   0.4378 |   0.459  |    0.479  |    0.4957 |
| raw_tags_v4_q/test    | 0.7912 | 0.8483 |  0.889  |  0.9066 |  0.7912 |  1.0411 |   1.0673 |   1.0727 |   0.7912 |   0.8148 |    0.8337 |    0.8407 |
| raw_tags_v4/test      | 0.7718 | 0.8598 |  0.8981 |  0.9092 |  0.7718 |  1.0383 |   1.0645 |   1.0684 |   0.7718 |   0.8132 |    0.8314 |    0.836  |
| raw_tags_v4_bow/test  | 0.6487 | 0.6312 |  0.6776 |  0.7181 |  0.6487 |  0.7985 |   0.8205 |   0.8297 |   0.6487 |   0.615  |    0.6345 |    0.6488 |