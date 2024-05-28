

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
| baseline/train           | 0.8227 | 0.8652 |  0.8929 |  0.9003 |  0.8227 |  1.0661 |   1.0887 |   1.0915 |   0.8227 |   0.8353 |    0.849  |    0.8521 |
| raw_tags_v5_zh/train     | 0.702  | 0.774  |  0.8136 |  0.8306 |  0.702  |  0.9424 |   0.9677 |   0.9726 |   0.702  |   0.7329 |    0.7512 |    0.7577 |
| raw_tags_v5_zh_bow/train | 0.5849 | 0.6246 |  0.6762 |  0.7167 |  0.5849 |  0.7482 |   0.7713 |   0.7807 |   0.5849 |   0.5873 |    0.6087 |    0.6232 |
| raw_tags_v5_en/train     | 0.7716 | 0.8575 |  0.8914 |  0.9023 |  0.7716 |  1.0305 |   1.0546 |   1.0581 |   0.7716 |   0.812  |    0.828  |    0.8323 |
| raw_tags_v5_en_bow/train | 0.6743 | 0.7073 |  0.7569 |  0.7873 |  0.6743 |  0.8661 |   0.8917 |   0.8999 |   0.6743 |   0.6778 |    0.6996 |    0.711  |
| keywords/train           | 0.5408 | 0.6677 |  0.7392 |  0.7762 |  0.5408 |  0.7591 |   0.7921 |   0.8021 |   0.5408 |   0.5983 |    0.6295 |    0.6434 |
| keywords_bow/train       | 0.4328 | 0.5137 |  0.5646 |  0.6055 |  0.4328 |  0.5846 |   0.6059 |   0.6151 |   0.4328 |   0.4597 |    0.4808 |    0.4952 |
| raw_tags_v4_q/train      | 0.7862 | 0.8393 |  0.8792 |  0.8992 |  0.7862 |  1.0207 |   1.045  |   1.0511 |   0.7862 |   0.8046 |    0.8227 |    0.8306 |
| raw_tags_v4/train        | 0.7579 | 0.8486 |  0.8873 |  0.9013 |  0.7579 |  1.0119 |   1.037  |   1.0416 |   0.7579 |   0.8002 |    0.8182 |    0.8238 |
| raw_tags_v4_bow/train    | 0.6121 | 0.6095 |  0.662  |  0.7    |  0.6121 |  0.7571 |   0.7804 |   0.7896 |   0.6121 |   0.5885 |    0.6103 |    0.6241 |
| baseline/val             | 0.8388 | 0.8647 |  0.8973 |  0.9053 |  0.8388 |  1.0944 |   1.1233 |   1.1265 |   0.8388 |   0.8394 |    0.8558 |    0.8593 |
| raw_tags_v5_zh/val       | 0.7325 | 0.7925 |  0.8337 |  0.85   |  0.7325 |  0.9894 |   1.0193 |   1.0245 |   0.7325 |   0.7563 |    0.7758 |    0.7822 |
| raw_tags_v5_zh_bow/val   | 0.6237 | 0.6425 |  0.6975 |  0.7449 |  0.6237 |  0.793  |   0.8191 |   0.8311 |   0.6237 |   0.6115 |    0.6344 |    0.6519 |
| raw_tags_v5_en/val       | 0.7906 | 0.8625 |  0.8963 |  0.9069 |  0.7906 |  1.0663 |   1.0944 |   1.0984 |   0.7906 |   0.822  |    0.8384 |    0.8429 |
| raw_tags_v5_en_bow/val   | 0.6924 | 0.7174 |  0.7708 |  0.8013 |  0.6924 |  0.8994 |   0.9293 |   0.9385 |   0.6924 |   0.6915 |    0.7151 |    0.727  |
| keywords/val             | 0.5746 | 0.6818 |  0.7501 |  0.7923 |  0.5746 |  0.8045 |   0.8397 |   0.8518 |   0.5746 |   0.6201 |    0.6502 |    0.6665 |
| keywords_bow/val         | 0.4797 | 0.5383 |  0.5889 |  0.6254 |  0.4797 |  0.6388 |   0.6627 |   0.6724 |   0.4797 |   0.4915 |    0.5127 |    0.5264 |
| raw_tags_v4_q/val        | 0.811  | 0.8377 |  0.8818 |  0.904  |  0.811  |  1.053  |   1.0822 |   1.0892 |   0.811  |   0.812  |    0.8321 |    0.841  |
| raw_tags_v4/val          | 0.784  | 0.8512 |  0.8913 |  0.907  |  0.784  |  1.0491 |   1.0784 |   1.0839 |   0.784  |   0.812  |    0.8309 |    0.8374 |
| raw_tags_v4_bow/val      | 0.6553 | 0.6274 |  0.68   |  0.7225 |  0.6553 |  0.8036 |   0.8302 |   0.8411 |   0.6553 |   0.6152 |    0.6377 |    0.6533 |
| baseline/test            | 0.8458 | 0.8751 |  0.904  |  0.9103 |  0.8458 |  1.0967 |   1.1205 |   1.1229 |   0.8458 |   0.8501 |    0.8643 |    0.867  |
| raw_tags_v5_zh/test      | 0.6966 | 0.7737 |  0.8092 |  0.8266 |  0.6966 |  0.9554 |   0.9798 |   0.9845 |   0.6966 |   0.7323 |    0.7489 |    0.7553 |
| raw_tags_v5_zh_bow/test  | 0.5819 | 0.6211 |  0.6745 |  0.7166 |  0.5819 |  0.7571 |   0.7832 |   0.7928 |   0.5819 |   0.5839 |    0.6071 |    0.6221 |
| raw_tags_v5_en/test      | 0.7882 | 0.8706 |  0.9035 |  0.9114 |  0.7882 |  1.0604 |   1.0852 |   1.0883 |   0.7882 |   0.8262 |    0.8421 |    0.8455 |
| raw_tags_v5_en_bow/test  | 0.6487 | 0.6888 |  0.7404 |  0.7704 |  0.6487 |  0.8563 |   0.8825 |   0.8906 |   0.6487 |   0.658  |    0.6803 |    0.6916 |
| keywords/test            | 0.5483 | 0.6811 |  0.7558 |  0.7928 |  0.5483 |  0.7885 |   0.8227 |   0.8325 |   0.5483 |   0.6111 |    0.6436 |    0.6575 |
| keywords_bow/test        | 0.4429 | 0.5037 |  0.552  |  0.6009 |  0.4429 |  0.6011 |   0.6225 |   0.6328 |   0.4429 |   0.4596 |    0.4799 |    0.4969 |
| raw_tags_v4_q/test       | 0.792  | 0.8481 |  0.8902 |  0.9081 |  0.792  |  1.0418 |   1.0682 |   1.0736 |   0.792  |   0.8154 |    0.8347 |    0.8418 |
| raw_tags_v4/test         | 0.7718 | 0.8597 |  0.8979 |  0.9101 |  0.7718 |  1.0384 |   1.0644 |   1.0685 |   0.7718 |   0.8131 |    0.8311 |    0.8361 |
| raw_tags_v4_bow/test     | 0.6496 | 0.6335 |  0.6801 |  0.7202 |  0.6496 |  0.8002 |   0.8225 |   0.8314 |   0.6496 |   0.6168 |    0.6365 |    0.6507 |