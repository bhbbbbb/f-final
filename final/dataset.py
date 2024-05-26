from typing import Literal
import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from . import data


class Dataset(TorchDataset):

    def __init__(
        self,
        split: Literal['train', 'val'],
        raw_tags: dict[int, list[str]],
        user_names: dict[int, str] = None,
        n_tags_per_product: int = 5,
        max_load_len: int = 20,
        predict_purchase_only: bool = True,
    ):
        super().__init__()
        self.df = pd.DataFrame(Dataset.get_expanded_dataset(split)
                               ).sample(frac=1)
        self.raw_tags = raw_tags
        if user_names is None:
            print('user_names is not used.')

            class DummyMapping:

                def __getitem__(self, _item):
                    return 'First time buyer'

            user_names = DummyMapping()
        self.user_names = user_names
        self.predict_purchase_only = predict_purchase_only
        self.n_tags_per_product = n_tags_per_product
        self.max_load_len = max_load_len
        return

    @staticmethod
    def get_expanded_dataset(split: Literal['train', 'val']):
        merged_df = data.merged_df()[split]
        for _order_id, row in merged_df.iterrows():
            for product in row['products']:
                yield {
                    'seq': row['loaded_pids'] + [product],
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

        seq, uuid = self.df.iloc[index]
        seq = seq[-self.max_load_len:]
        seq = list(map(pid_to_input, range(len(seq)), seq))
        if self.predict_purchase_only:
            data = {
                'instruction': f'User: {self.user_names[uuid]}',
                'input': '\n'.join(seq[:-1]),
                'output': seq[-1],
            }
        else:
            data = {
                'instruction': f'User: {self.user_names[uuid]}',
                'input': seq[0],
                'output': '\n'.join(seq[1:]),
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
    ):
        super().__init__()
        self.user_names = user_names
        self.raw_tags = raw_tags
        self.n_tags_per_product = n_tags_per_product
        return

    def __len__(self):
        return data.N_RPODUCTS

    def __getitem__(self, index: int):

        assert self.user_names is None

        def pid_to_input(i: int, pid: int):
            return (
                f'{i+1}. ' +
                ', '.join(self.raw_tags[pid][:self.n_tags_per_product])
            )

        input_ = pid_to_input(0, data.AVAILABLE_PRODUCT_IDS[index])
        return {
            'instruction': 'User: First time buyer',
            'input': input_,
        }
