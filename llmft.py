import os
import sys
import argparse
import json
import warnings

import torch.utils.data

warnings.filterwarnings("ignore")

import torch

import final
from final.llm import LLM, PROMPT


def main():
    PREDICT_PURCHASE_ONLY = False
    PREDICT_LOADED_ONLY = True
    N_TAGS_PER_PRODUCT = 7
    MAX_LOAD_LEN = 7
    product_raw_tags = final.data.product_tags_v5_en()
    train_dataset = final.Dataset(
        'train+val',
        raw_tags=product_raw_tags,
        # user_names=final.data.default_user_names(),
        predict_purchase_only=PREDICT_PURCHASE_ONLY,
        n_tags_per_product=N_TAGS_PER_PRODUCT,
        max_load_len=MAX_LOAD_LEN,
        predict_loaded_only=PREDICT_LOADED_ONLY,
    )
    val_dataset = final.Dataset(
        'val',
        raw_tags=product_raw_tags,
        # user_names=final.data.default_user_names(),
        predict_purchase_only=PREDICT_PURCHASE_ONLY,
        n_tags_per_product=N_TAGS_PER_PRODUCT,
        max_load_len=MAX_LOAD_LEN,
        predict_loaded_only=PREDICT_LOADED_ONLY,
    )
    INSTRUCTION = (
        'Predict the next product tags based on browsing histories. '
        f'Provide exactly {N_TAGS_PER_PRODUCT} output tags, comma-separated, sorted by relevance. '
    )
    print(INSTRUCTION)
    for i in range(3):
        d = train_dataset[i]
        print(
            PROMPT.format(
                instruction=INSTRUCTION,
                user_instruction=d['instruction'],
                input=d['input'],
            ) + ' ' + d['output']
        )
    print(f'{len(train_dataset) = }')
    llm = LLM(
        # model_name='./dataset/taide_7b',
        # model_name='llama-2-7b',
        model_name='togethercomputer/Llama-2-7B-32K-Instruct',
        instruction=INSTRUCTION,
        ckpt_dir='demos/exp2',
        real_ckpt_dir='ckpts',
        inference_only=False,
    )

    llm.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_total_limit=10,
        save_steps=100,
        num_epoch=1,
        learning_rate=3e-5,
    )

    # with open(
    #     os.path.join(llm.ckpt_name, 'inference.json'), 'w', encoding='utf8'
    # ) as fout:
    #     json.dump(results, fout, ensure_ascii=False)


main()
