import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import final


def get_enhanced_tags_metrics():
    RAW_PREDICT = 'results/ckpt-14/'
    with open(
        os.path.join(RAW_PREDICT, 'enhanced_tags.json'), 'r', encoding='utf8'
    ) as fin:
        enhanced_tags = json.load(fin)
        enhanced_tags = {int(pid): v for pid, v in enhanced_tags.items()}

    result_metrics = {}

    for split, merged_df in final.data.merged_df().items():
        merged_df = final.utils.next_product_expansion(merged_df)
        result_metrics[split] = final.TagsEvaluator(
            enhanced_tags, query_fn='union'
        ).cal_ranking_metrics(merged_df)
    return result_metrics


def main():
    result_df = pd.read_csv('results/sas4rec.csv')
    dfs = {
        split: result_df[result_df['split'] == split][:100]
        for split in ['train', 'val', 'test']
    }

    METRIC = 'ndcg@20'
    val_df = dfs['val']
    print(val_df.iloc[-1])
    print(val_df[val_df[METRIC] == val_df[METRIC].max()])


def plot():

    result_df = pd.read_csv('results/sas4rec.csv')

    SPLITS = ['train', 'val', 'test']
    MAX_EPOCHS = 300 // 5
    dfs = {
        split: result_df[result_df['split'] == split][:MAX_EPOCHS]
        for split in SPLITS
    }

    METRIC = 'ndcg@20'
    enhanced_tags_metrics = get_enhanced_tags_metrics()

    colors = {
        'train': 'blue',
        'val': 'orange',
        'test': 'green',
    }
    for split, df in dfs.items():
        plt.plot(
            df['epoch'],
            df[METRIC],
            label=f'{split}/SASRec',
            color=colors[split],
        )
        plt.plot(
            df['epoch'],
            [enhanced_tags_metrics[split][METRIC]] * len(df['epoch']),
            label=f'{split}/EnhancedTags',
            color=colors[split],
            linestyle='dashed',
        )

    plt.legend()
    plt.title('NDCG@20')
    plt.savefig('figs/sasrec')
    plt.close()


if __name__ == '__main__':
    main()
