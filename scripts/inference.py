import json
import os
import final
from final.llm import LLM
from transformers import GenerationConfig
from tqdm import tqdm
import hashlib


def main():
    N_TAGS_PER_PRODUCT = 7
    INSTRUCTION = (
        'Predict the next product tags based on browsing histories. '
        f'Provide exactly {N_TAGS_PER_PRODUCT} output tags, comma-separated, sorted by relevance. '
    )
    PRODUCT_RAW_TAGS = final.data.product_tags_v5_en()

    inference_dataset = final.InferenceDataset(
        raw_tags=PRODUCT_RAW_TAGS,
        # user_names=final.data.default_user_names(),
        n_tags_per_product=N_TAGS_PER_PRODUCT,
        legacy_output_first_time_buyer=False,
    )
    # print(PRODUCT_RAW_TAGS)
    print(INSTRUCTION)
    print(inference_dataset[0])
    print(inference_dataset[1])
    print(inference_dataset[3])
    print(inference_dataset[5])

    llm = LLM(
        model_name='togethercomputer/Llama-2-7B-32K-Instruct',
        instruction=INSTRUCTION, ckpt_name='ckpts/ckpt-14',
        real_ckpt_dir='ckpts'
    )

    max_len = 128  # not sure
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=2.,  # important
        num_beams=1,
        top_p=0.3,  # not sure
        no_repeat_ngram_size=5,  # important (higher less diversity)
        pad_token_id=2,
    )
    # generation_config = GenerationConfig(
    #     do_sample=False,
    #     top_k=1,
    #     temperature=1.,  # important
    #     num_beams=1,
    #     top_p=0.3,  # not sure
    #     no_repeat_ngram_size=5,  # important (higher less diversity)
    #     pad_token_id=2,
    # )
    sha256 = hashlib.sha256(str(generation_config.to_dict()).encode()
                            ).hexdigest()
    log_dir = f'results/ckpt-{llm.cur_ckpt_count}/{sha256[:8]}'
    print(log_dir)
    single_inference_results = {}
    pbar = tqdm(
        enumerate(final.AVAILABLE_PRODUCT_IDS),
        total=len(final.AVAILABLE_PRODUCT_IDS)
    )
    for i, pid in pbar:
        d = inference_dataset[i]
        predict = llm.single_inference(
            d, generation_config, max_len, verbose=False
        )
        inputs_ = '\n'.join(d['input'].split('\n')[:-1])
        if i < 10:
            print(f'inputs{i}:\n{d["instruction"]}\n{inputs_}。\n', file=pbar)
            print(f'outputs:\n' + predict, file=pbar)
        single_inference_results[pid] = predict
    pbar.close()
    # assert set(single_inference_results) == set(final.AVAILABLE_PRODUCT_IDS)
    os.makedirs(log_dir, exist_ok=True)
    with open(
        os.path.join(log_dir, 'generation_config.json'), 'w', encoding='utf8'
    ) as fout:
        json.dump(generation_config.to_dict(), fout, indent=4)
    with open(
        os.path.join(log_dir, 'raw_predict.json'), 'w', encoding='utf8'
    ) as fout:
        json.dump(single_inference_results, fout, ensure_ascii=False, indent=4)
    print(log_dir)


# single_inference_results
if __name__ == '__main__':
    main()
