

## Install and Setup


```sh
cd final
pip install .
```

You may also need to install some additional dependencies. For working on colab, please follow the first cell of [llm_inference.ipynb](./demos/llm_inference.ipynb). Otherwise, check the `requirements.txt`.

Check [general_demo](./demos/demo.ipynb) for more demonstration of basic usage.

## `final.data`

load the cleaned and preprocessed data automatically (data would be downloaded and cached in directory `dataset`).

check [demo](./demos/data.ipynb) for more details.

Note that some data are encrypted, you need to set the password via `final.data.set_password` 


## LLM Finetuning

- Check the [script](./scripts/llmft.py) for finetuning

## LLM Inference

Checkpoints are available [here](https://www.dropbox.com/scl/fo/fi3u49a9n4bsoc7lmj9f4/AIYGHFGbfRjniaUe-0tztBc?rlkey=rr709y1afqqlwf8xznzg2l4tn&st=1rjwy0up&dl=0).

- Check the [ipynb](./demos/llm_inference.ipynb) for inference and evaluation.

## Visualization

Check the following scripts

- [heatmap](./scripts/heatmap_tags.py)
- [eda](./scripts/eda.py)

---

## Results

- `baseline`: score of a product is its highest index in the load sequence.
- `keyword`: keywords directly from i3fresh
- `raw_tags_v4`: raw tags from `Huan/ver4.csv`
- `raw_tags_v5_zh`: raw tags from `Huan/ver5_zh.csv`
- `raw_tags_v5_en`: raw tags from `Huan/ver5_en.csv`
- `*_bow`: tags/keywords preprocessed using bag-of-words.
    - If the lang=zh, a chinese character is considered a word



| name                 |   hr@5 |   hr@10 |   hr@20 |   mrr@5 |   mrr@10 |   mrr@20 |   ndcg@5 |   ndcg@10 |   ndcg@20 |
|:---------------------|-------:|--------:|--------:|--------:|---------:|---------:|---------:|----------:|----------:|
| baseline/train       | 0.2073 |  0.2296 |  0.2579 |  0.117  |   0.1201 |   0.122  |   0.1399 |    0.1472 |    0.1543 |
| keywords/train       | 0.2034 |  0.2676 |  0.3183 |  0.1239 |   0.1326 |   0.1361 |   0.1436 |    0.1644 |    0.1773 |
| raw_tags_v5_en/train | 0.2475 |  0.2957 |  0.3409 |  0.1535 |   0.1599 |   0.1631 |   0.177  |    0.1926 |    0.2041 |
| enhanced_tags/train  | 0.2606 |  0.3156 |  0.3637 |  0.1572 |   0.1646 |   0.168  |   0.1831 |    0.2009 |    0.2131 |
| baseline/val         | 0.206  |  0.2341 |  0.2618 |  0.1212 |   0.125  |   0.1269 |   0.1427 |    0.1518 |    0.1589 |
| keywords/val         | 0.2147 |  0.2811 |  0.3277 |  0.1361 |   0.145  |   0.1482 |   0.1557 |    0.1772 |    0.1889 |
| raw_tags_v5_en/val   | 0.2555 |  0.3044 |  0.348  |  0.1573 |   0.164  |   0.167  |   0.1817 |    0.1977 |    0.2087 |
| enhanced_tags/val    | 0.2535 |  0.3093 |  0.3529 |  0.1533 |   0.1609 |   0.164  |   0.1783 |    0.1966 |    0.2076 |
| baseline/test        | 0.2194 |  0.2547 |  0.281  |  0.1191 |   0.124  |   0.1258 |   0.1444 |    0.156  |    0.1627 |
| keywords/test        | 0.2201 |  0.293  |  0.3403 |  0.1327 |   0.1423 |   0.1457 |   0.1544 |    0.1779 |    0.1899 |
| raw_tags_v5_en/test  | 0.2652 |  0.3231 |  0.3629 |  0.1617 |   0.1694 |   0.1721 |   0.1876 |    0.2062 |    0.2162 |
| enhanced_tags/test   | 0.281  |  0.3426 |  0.3884 |  0.1615 |   0.1699 |   0.1731 |   0.1913 |    0.2114 |    0.2231 |
