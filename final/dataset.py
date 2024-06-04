import re
from typing import Literal
import pandas as pd
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from . import data, utils


class Dataset(TorchDataset):

    def __init__(
        self,
        split: Literal['train', 'val', 'train+val'],
        raw_tags: dict[int, list[str]],
        user_names: dict[int, str] = None,
        n_tags_per_product: int = 5,
        max_load_len: int = 20,
        predict_purchase_only: bool = False,
        predict_loaded_only: bool = False,
    ):
        super().__init__()
        self.df = pd.DataFrame(
            Dataset.get_expanded_dataset(
                split,
                expand=predict_purchase_only,
                loaded_only=predict_loaded_only,
            )
        ).sample(frac=1)
        self.raw_tags = raw_tags
        self.user_names = user_names
        self.predict_purchase_only = predict_purchase_only
        self.n_tags_per_product = n_tags_per_product
        self.max_load_len = max_load_len
        return

    @staticmethod
    def get_expanded_dataset(
        split: Literal['train', 'val', 'train+val'],
        expand: bool = True,
        loaded_only: bool = None,
    ):
        if split == 'train+val':
            merged_df = pd.concat(
                [data.merged_df()['train'],
                 data.merged_df()['val']], axis=0
            )
        else:
            merged_df = data.merged_df()[split]
        if loaded_only:
            merged_df = merged_df[merged_df['loaded_pids'].map(len) >= 2]
        for _order_id, row in merged_df.iterrows():
            if expand:
                for product in row['products']:
                    yield {
                        'seq': row['loaded_pids'] + [product],
                        'uuid': row['uuid_ind'],
                    }
            elif loaded_only is False:
                purchased_products = list(row['products'])
                np.random.shuffle(purchased_products)
                yield {
                    'seq': row['loaded_pids'] + purchased_products,
                    'uuid': row['uuid_ind'],
                }
            else:
                assert loaded_only is True
                yield {
                    'seq': row['loaded_pids'],
                    'uuid': row['uuid_ind'],
                }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):

        def pid_to_input(i: int, pid: int):
            return (
                f'{i+1}. ' +
                ', '.join(self.raw_tags[pid][:self.n_tags_per_product])
            )

        def de_numbering(output: str):
            return re.sub(r'\d+\. ', '', output)

        seq, uuid = self.df.iloc[index]
        seq = seq[-self.max_load_len:]
        seq = list(map(pid_to_input, range(len(seq)), seq))
        if self.predict_purchase_only:
            data = {
                'instruction': '' if self.user_names is None else
                f'User: {self.user_names[uuid]}',
                'input': '\n'.join(seq[:-1]) + f'\n{len(seq)}.',
                'output': de_numbering(seq[-1]),
            }
        else:
            assert len(seq) >= 2, str(seq)
            if len(seq[1:]) == 1:
                output_ = de_numbering(seq[1])
            else:
                output_ = '\n'.join([de_numbering(seq[1]), *seq[2:]])

            data = {
                'instruction': '' if self.user_names is None else
                f'User: {self.user_names[uuid]}',
                'input': seq[0] + f'\n2.',
                'output': output_
            }
        return data

    # def __iter__(self):
    #     info = get_worker_info()
    #     if info is None:
    #         n = self.samples_per_epoch
    #     else:
    #         n = self.samples_per_epoch // info.num_workers

    #     for i in self.range(n):
    #         order = np.random.choice(self.orders, 1)[0]
    #         sample = np.random.choice(order, 1)[0]
    #         yield sample


class InferenceDataset(TorchDataset):

    def __init__(
        self,
        raw_tags: dict[int, list[str]],
        user_names: dict[int, str] = None,
        n_tags_per_product: int = 5,
        legacy_output_first_time_buyer: bool = False,
    ):
        super().__init__()
        self.user_names = user_names
        self.raw_tags = raw_tags
        self.n_tags_per_product = n_tags_per_product
        self.output_first_time_buyer = legacy_output_first_time_buyer
        assert legacy_output_first_time_buyer is False
        return

    def __len__(self):
        return data.N_RPODUCTS

    def __getitem__(self, index: int):

        assert self.user_names is None

        pid = data.AVAILABLE_PRODUCT_IDS[index]
        input_ = (
            '1. ' + ', '.join(self.raw_tags[pid][:self.n_tags_per_product]) +
            '\n2.'
        )

        return {
            'instruction':
            'User: First time buyer' if self.output_first_time_buyer else '',
            'input':
            input_,
        }


class TestDataset(TorchDataset):

    def __init__(
        self,
        raw_tags: dict[int, list[str]],
        user_names: dict[int, str] = None,
        n_tags_per_product: int = 5,
        max_load_len: int = 20,
        *args,
        **kwargs,
    ):
        super().__init__()
        merged_df = data.merged_df()['test']
        self.df = utils.next_product_expansion(merged_df)
        self.raw_tags = raw_tags
        self.user_names = user_names
        assert user_names is None
        self.n_tags_per_product = n_tags_per_product
        self.max_load_len = max_load_len
        return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):

        def pid_to_input(i: int, pid: int):
            return (
                f'{i+1}. ' +
                ', '.join(self.raw_tags[pid][:self.n_tags_per_product])
            )

        seq = self.df['loaded_pids'].iloc[index]
        seq = seq[-self.max_load_len:]
        seq = list(map(pid_to_input, range(len(seq)), seq))

        data = {
            'instruction': '',
            'input': '\n'.join(seq) + f'\n{len(seq) + 1}.',
        }
        return data
