

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
- `raw_tags_v5_zh`: raw tags from `Huan/ver5_zh.csv`
- `raw_tags_v5_en`: raw tags from `Huan/ver5_en.csv`
- `*_bow`: tags/keywords preprocessed using bag-of-words.
    - If the lang=zh, a chinese character is considered a word



| name                     |   hr@1 |   hr@5 |   hr@10 |   hr@20 |   mrr@1 |   mrr@5 |   mrr@10 |   mrr@20 |   ndcg@1 |   ndcg@5 |   ndcg@10 |   ndcg@20 |
|:-------------------------|-------:|-------:|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|----------:|----------:|
| baseline/train           | 0.8227 | 0.8651 |  0.893  |  0.9001 |  0.8227 |  1.066  |   1.0886 |   1.0914 |   0.8227 |   0.8352 |    0.8489 |    0.8519 |
| raw_tags_v5_zh/train     | 0.7025 | 0.774  |  0.8132 |  0.8297 |  0.7025 |  0.9424 |   0.9676 |   0.9725 |   0.7025 |   0.7329 |    0.7511 |    0.7574 |
| raw_tags_v5_zh_bow/train | 0.5844 | 0.6249 |  0.6771 |  0.7175 |  0.5844 |  0.748  |   0.7712 |   0.7806 |   0.5844 |   0.5872 |    0.6088 |    0.6233 |
| raw_tags_v5_en/train     | 0.6303 | 0.6839 |  0.7387 |  0.7687 |  0.6303 |  0.8214 |   0.8469 |   0.8546 |   0.6303 |   0.6437 |    0.6668 |    0.6779 |
| raw_tags_v5_en_bow/train | 0.6301 | 0.6844 |  0.7384 |  0.768  |  0.6301 |  0.8215 |   0.8468 |   0.8545 |   0.6301 |   0.6439 |    0.6668 |    0.6777 |
| keywords/train           | 0.5434 | 0.6675 |  0.7383 |  0.7758 |  0.5434 |  0.76   |   0.793  |   0.803  |   0.5434 |   0.5986 |    0.6296 |    0.6436 |
| keywords_bow/train       | 0.4301 | 0.5136 |  0.5632 |  0.6042 |  0.4301 |  0.5831 |   0.6043 |   0.6135 |   0.4301 |   0.4586 |    0.4791 |    0.4936 |
| raw_tags_v4_q/train      | 0.7856 | 0.8394 |  0.8789 |  0.8989 |  0.7856 |  1.0203 |   1.0445 |   1.0506 |   0.7856 |   0.8045 |    0.8224 |    0.8303 |
| raw_tags_v4/train        | 0.7566 | 0.8486 |  0.8875 |  0.901  |  0.7566 |  1.0112 |   1.0364 |   1.0409 |   0.7566 |   0.7998 |    0.8179 |    0.8234 |
| raw_tags_v4_bow/train    | 0.6119 | 0.6093 |  0.6623 |  0.7008 |  0.6119 |  0.7568 |   0.7803 |   0.7895 |   0.6119 |   0.5883 |    0.6103 |    0.6242 |
| baseline/val             | 0.8388 | 0.8646 |  0.8967 |  0.9041 |  0.8388 |  1.0944 |   1.1228 |   1.1259 |   0.8388 |   0.8393 |    0.8555 |    0.8587 |
| raw_tags_v5_zh/val       | 0.7317 | 0.7919 |  0.8328 |  0.8488 |  0.7317 |  0.9886 |   1.0185 |   1.0235 |   0.7317 |   0.7559 |    0.7753 |    0.7816 |
| raw_tags_v5_zh_bow/val   | 0.6237 | 0.6428 |  0.6961 |  0.7459 |  0.6237 |  0.793  |   0.8188 |   0.831  |   0.6237 |   0.6117 |    0.6341 |    0.6522 |
| raw_tags_v5_en/val       | 0.652  | 0.6899 |  0.7462 |  0.7809 |  0.652  |  0.8523 |   0.8817 |   0.891  |   0.652  |   0.6538 |    0.678  |    0.6911 |
| raw_tags_v5_en_bow/val   | 0.6509 | 0.688  |  0.7458 |  0.7804 |  0.6509 |  0.8503 |   0.88   |   0.8893 |   0.6509 |   0.652  |    0.6767 |    0.6898 |
| keywords/val             | 0.5781 | 0.6844 |  0.7525 |  0.7973 |  0.5781 |  0.8072 |   0.8425 |   0.8548 |   0.5781 |   0.6224 |    0.6525 |    0.6695 |
| keywords_bow/val         | 0.481  | 0.5383 |  0.588  |  0.6251 |  0.481  |  0.6401 |   0.6638 |   0.6736 |   0.481  |   0.4926 |    0.5135 |    0.5275 |
| raw_tags_v4_q/val        | 0.8102 | 0.8376 |  0.8815 |  0.904  |  0.8102 |  1.0528 |   1.0819 |   1.089  |   0.8102 |   0.8117 |    0.8317 |    0.8407 |
| raw_tags_v4/val          | 0.7821 | 0.8507 |  0.8919 |  0.9078 |  0.7821 |  1.0479 |   1.0772 |   1.0829 |   0.7821 |   0.8113 |    0.8305 |    0.8371 |
| raw_tags_v4_bow/val      | 0.6556 | 0.6276 |  0.681  |  0.722  |  0.6556 |  0.8042 |   0.8308 |   0.8416 |   0.6556 |   0.6154 |    0.6382 |    0.6535 |
| baseline/test            | 0.8454 | 0.8744 |  0.9036 |  0.9109 |  0.8454 |  1.0962 |   1.1201 |   1.1226 |   0.8454 |   0.8495 |    0.8639 |    0.8668 |
| raw_tags_v5_zh/test      | 0.695  | 0.7734 |  0.8081 |  0.8224 |  0.695  |  0.9549 |   0.9791 |   0.9835 |   0.695  |   0.7314 |    0.7476 |    0.7532 |
| raw_tags_v5_zh_bow/test  | 0.5803 | 0.6214 |  0.6766 |  0.7181 |  0.5803 |  0.7558 |   0.7822 |   0.7915 |   0.5803 |   0.5834 |    0.6072 |    0.6219 |
| raw_tags_v5_en/test      | 0.6059 | 0.6589 |  0.7143 |  0.7493 |  0.6059 |  0.8015 |   0.8278 |   0.8363 |   0.6059 |   0.619  |    0.6427 |    0.6554 |
| raw_tags_v5_en_bow/test  | 0.605  | 0.6583 |  0.712  |  0.7476 |  0.605  |  0.8006 |   0.827  |   0.8358 |   0.605  |   0.6179 |    0.6412 |    0.6543 |
| keywords/test            | 0.5576 | 0.6801 |  0.756  |  0.7918 |  0.5576 |  0.7933 |   0.8277 |   0.8374 |   0.5576 |   0.6131 |    0.646  |    0.6595 |
| keywords_bow/test        | 0.437  | 0.5072 |  0.5533 |  0.6015 |  0.437  |  0.5994 |   0.6203 |   0.6304 |   0.437  |   0.4592 |    0.4787 |    0.4953 |
| raw_tags_v4_q/test       | 0.7899 | 0.8481 |  0.8889 |  0.9075 |  0.7899 |  1.0405 |   1.0667 |   1.0722 |   0.7899 |   0.8145 |    0.8334 |    0.8407 |
| raw_tags_v4/test         | 0.7727 | 0.8608 |  0.8992 |  0.9111 |  0.7727 |  1.0396 |   1.0654 |   1.0696 |   0.7727 |   0.8139 |    0.8319 |    0.8369 |
| raw_tags_v4_bow/test     | 0.6483 | 0.6315 |  0.6799 |  0.7203 |  0.6483 |  0.7985 |   0.821  |   0.8301 |   0.6483 |   0.6151 |    0.6354 |    0.6497 |