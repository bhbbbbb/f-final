import os
import sys
import warnings
import logging

import torch.utils
import torch.utils.data

warnings.filterwarnings("ignore")

import torch
import transformers
from peft import PeftModel
from colorama import *

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GenerationConfig
from peft import (
    prepare_model_for_int8_training, LoraConfig, get_peft_model,
    get_peft_model_state_dict, prepare_model_for_kbit_training
)


class _WrapDataset(torch.utils.data.Dataset):

    def __init__(self, ref_dataset: torch.utils.data.Dataset, hook):
        self.ref = ref_dataset
        self.hook = hook
        return

    def len(self):
        return len(self.ref)

    def __getitem__(self, index):
        return self.hook(self.ref[index])


LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"
]
LEARNING_RATE = 3e-4
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
CUTOFF_LEN = 256
VAL_SET_SIZE = 0


class LLM:

    def __init__(
        self,
        model_name: str,
        instruction: str,
        ckpt_dir: str,
        ckpt_name: str = None,
        cache_dir: str = './cache',
    ):
        self.model_name = model_name
        self.instruction = instruction
        self.cache_dir = cache_dir
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self._init(model_name)
        return

    def _init(self, model_name: str):

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            quantization_config=nf4_config,
            # low_cpu_mem_usage=True
        )

        logging.getLogger('transformers').setLevel(logging.ERROR)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            cache_dir=self.cache_dir,
            quantization_config=nf4_config,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.ckpt_name is not None:
            print(f'loading checkpoint from {self.ckpt_name}')
            self.model = PeftModel.from_pretrained(self.model, self.ckpt_name)
        else:
            print('Warning: checkpoint not used.')
        return

    def train(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        num_epoch: int,
        save_steps: int = 15,
        save_total_limit=3,  # maximux saved ckpts
        learning_rate: float = LEARNING_RATE,
        report_to=None,
    ):
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = prepare_model_for_int8_training(self.model)

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        self.tokenizer.pad_token_id = 0

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=_WrapDataset(
                train_dataset, self._generate_training_data
            ),
            eval_dataset=_WrapDataset(
                val_dataset, self._generate_training_data
            ),
            args=transformers.TrainingArguments(
                per_device_train_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=50,
                num_train_epochs=num_epoch,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=save_steps,
                save_strategy="steps",
                save_steps=save_steps,
                output_dir=self.ckpt_dir,
                save_total_limit=save_total_limit,
                ddp_find_unused_parameters=None,
                report_to=report_to,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        self.model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != 'win32':
            self.model = torch.compile(self.model)

        trainer.train()

        self.model.save_pretrained(self.ckpt_dir)
        return

    # def inference(self):
    #     nf4_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )

    #     tokenizer = AutoTokenizer.from_pretrained(
    #         model_name,
    #         cache_dir=cache_dir,
    #         quantization_config=nf4_config,
    #     )

    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         quantization_config=nf4_config,
    #         device_map={'': 0},
    #         cache_dir=cache_dir
    #     )

    #     model = PeftModel.from_pretrained(model, ckpt_name, device_map={'': 0})

    #     results = []

    #     generation_config = GenerationConfig(
    #         do_sample=True,
    #         temperature=temperature,
    #         num_beams=1,
    #         top_p=top_p,
    #         # top_k=top_k,
    #         no_repeat_ngram_size=3,
    #         pad_token_id=2
    #     )

    def _generate_training_data(self, data_point):
        """
        (1) Goal:
            - This function is used to transform a data point (input and output texts) to tokens that our model can read

        (2) Arguments:
            - data_point: dict, with field "instruction", "input", and "output" which are all str

        (3) Returns:
            - a dict with model's input tokens, attention mask that make our model causal, and corresponding output targets

        (3) Example:
            - If you construct a dict, data_point_1, with field "instruction", "input", and "output" which are all str, you can use the function like this:
                formulate_article(data_point_1)

        """
        # construct full input prompt
        prompt = f"""\
[INST] <<SYS>>
{self.instruction}
<</SYS>>

{data_point['instruction']}
{data_point['input']}
[/INST]"""
        # count the number of input tokens
        len_user_prompt_tokens = (
            len(
                self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=CUTOFF_LEN + 1,
                    padding="max_length",
                )["input_ids"]
            ) - 1
        )
        # transform input prompt into tokens
        full_tokens = self.tokenizer(
            prompt + " " + data_point["output"] + "</s>",
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids":
            full_tokens,
            "labels": [-100] * len_user_prompt_tokens +
            full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    def single_inference(
        self,
        data_point,
        generation_config,
        max_len,
        verbose=False,
    ):
        """
        (1) Goal:
            - This function is used to get the model's output given input strings

        (2) Arguments:
            - instruction: str, description of what you want model to do
            - generation_config: transformers.GenerationConfig object, to specify decoding parameters relating to model inference
            - max_len: int, max length of model's output
            - input: str, input string the model needs to solve the instruction, default is "" (no input)
            - verbose: bool, whether to print the mode's output, default is True

        (3) Returns:
            - output: str, the mode's response according to the instruction and the input

        (3) Example:
            - If you the instruction is "ABC" and the input is "DEF" and you want model to give an answer under 128 tokens, you can use the function like this:
                evaluate(instruction="ABC", generation_config=generation_config, max_len=128, input="DEF")

        """
        prompt = f"""\
[INST] <<SYS>>
{self.instruction}
<</SYS>>

{data_point['instruction']}
{data_point['input']}
[/INST]"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()

        generation_output = self.model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_len,
        )

        for s in generation_output.sequences:
            raw_output = self.tokenizer.decode(s)
            output = raw_output.split("[/INST]")[1].replace(
                "</s>", ""
            ).replace("<s>", "").replace("Assistant:",
                                         "").replace("Assistant", "").strip()
            if (verbose):
                print('raw_output: ', raw_output)
                print('output: ', output)

        return output

    def iterative_inference(
        self,
        data_point,
        generation_config,
        max_len,
        n: int,
        verbose=False,
    ):

        assert n > 1
        data_point = dict(data_point)
        for _ in range(n):
            predict = self.single_inference(
                data_point,
                generation_config,
                max_len,
                verbose=verbose,
            )
            data_point['input'] += f'\n{predict}'
        return data_point['input']

    def multi_beam_inference(
        self,
        data_point,
        generation_config,
        max_len,
        n: int,
        verbose=False,
    ):

        assert n > 1
        predicts = data_point['input']
        for _ in range(n):
            predict = self.single_inference(
                data_point,
                generation_config,
                max_len,
                verbose=verbose,
            )
            predicts += f'\n{predict}'
        return predicts
