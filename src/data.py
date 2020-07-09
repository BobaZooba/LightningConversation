import os
import math
import pickle
from numpy import random
import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from typing import List, Optional, Tuple


class BatchPreparing:

    def __init__(self,
                 sep_index: int,
                 context_index: Optional[int] = None,
                 pad_index: int = 0):

        self.sep_index = sep_index
        self.context_index = context_index
        self.pad_index = pad_index

        self.response_segment_index = 1
        self.query_segment_index = 2
        self.context_segment_index = 3 if self.context_index is not None else None

    def padding(self, sequence: List[int], max_length: int) -> List[int]:

        sequence = sequence[:max_length]

        sequence += [self.pad_index] * (max_length - len(sequence))

        return sequence

    def get_segment_indices_with_context(self, sequence: List[int]) -> List[int]:

        last_context_index = len(sequence) - sequence[::-1].index(self.context_index) - 1
        output_indices = [self.context_segment_index] * (last_context_index + 1)
        output_indices += [self.query_segment_index] * (len(sequence) - last_context_index - 1)

        return output_indices

    def get_position_indices_with_context(self, sequence: List[int]) -> List[int]:

        time_stamps = list()
        current_time_stamp = 0

        for i in sequence:
            if i == self.context_index:
                current_time_stamp = 0
            else:
                current_time_stamp += 1
            time_stamps.append(current_time_stamp)

        return time_stamps

    def prepare_batch(self, batch: List[List[int]]) -> Tuple:

        source_sequence = []
        target_sequence = []
        segment_indices = []
        position_indices = []

        max_length = max([len(sample) for sample in batch])

        for sample in batch:
            sequence = self.padding(sample, max_length=max_length)

            source_sequence.append(sequence[:-1])
            target_sequence.append(sequence[1:])

            sample_sep_index = sample.index(self.sep_index)
            pad_part = [self.pad_index] * (max_length - len(sample))

            if self.context_index is not None and self.context_index in sequence:
                query_sequence = sequence[:sample_sep_index]
                query_segment = self.get_segment_indices_with_context(query_sequence)
                query_position = self.get_position_indices_with_context(query_sequence)
            else:
                query_segment = [self.query_segment_index] * sample_sep_index
                query_position = list(range(sample_sep_index))

            response_segment = [self.response_segment_index] * (len(sample) - sample_sep_index)
            response_position = list(range(len(sample) - sample_sep_index))

            segment = query_segment + response_segment + pad_part
            position = query_position + response_position + pad_part

            segment_indices.append(segment[:len(sequence[:-1])])
            position_indices.append(position[:len(sequence[:-1])])

        source_sequence = torch.tensor(source_sequence).long()
        target_sequence = torch.tensor(target_sequence).long()
        segment_indices = torch.tensor(segment_indices).long()
        position_indices = torch.tensor(position_indices).long()

        return source_sequence, target_sequence, segment_indices, position_indices

    def collate(self, batch: List[List[List[int]]]) -> Tuple:
        return self.prepare_batch(batch[0])


class BatchingStrategy:

    def __init__(self,
                 batching_type: str = 'sb',
                 batch_size: int = 64,
                 max_length: int = 64,
                 pre_shuffle_sort: bool = False):

        self.batching_type = batching_type
        self.batch_size = batch_size
        self.max_length = max_length
        self.pre_shuffle_sort = pre_shuffle_sort

        if self.batching_type in ['common_batching', 'common', 'classic', 'cmn']:
            self.batching_function = self.common_batching
        elif self.batching_type in ['sequence_bucketing', 'sb']:
            self.batching_function = self.sequence_bucketing
        elif self.batching_type in ['dynamic_batching', 'db']:
            self.batching_function = self.dynamic_batching
        else:
            raise ValueError('Available split types: common_batching, sequence_bucketing and dynamic_batching')

    def load_data(self, data_dir: str, num_workers: int = 1):

        data = list()

        if num_workers == -1:
            num_workers = mp.cpu_count()
        else:
            num_workers = num_workers

        chunk_paths = [os.path.join(data_dir, chunk_file)
                       for chunk_file in os.listdir(data_dir)
                       if chunk_file.startswith('chunk')
                       and chunk_file.endswith('pkl')]

        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                result = pool.map(self.load_file, chunk_paths)
            for batches in result:
                data.extend(batches)
        else:
            for file_path in chunk_paths:
                data.extend(self.load_file(file_path))

        return data

    def load_file(self, file_path: str) -> List[List[List[int]]]:
        with open(file_path, mode='rb') as file_object:
            data = pickle.load(file_object)

        data = [sample for sample in data if len(sample) <= self.max_length]

        batches = self.batching_function(data)

        return batches

    def common_batching(self, data: List[List[int]]) -> List[List[List[int]]]:

        if self.pre_shuffle_sort:
            random.shuffle(data)

        batches = [data[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
                   for i_batch in range(math.ceil(len(data) / self.batch_size))]

        random.shuffle(batches)

        return batches

    def sequence_bucketing(self, data: List[List[int]]) -> List[List[List[int]]]:

        if self.pre_shuffle_sort:
            random.shuffle(data)
            data = sorted(data, key=len)

        batches = [data[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]
                   for i_batch in range(math.ceil(len(data) / self.batch_size))]

        random.shuffle(batches)

        return batches

    def dynamic_batching(self, data: List[List[int]]) -> List[List[List[int]]]:
        """
        Dynamic batch size for high utilization (more than 95% on single gpu) with sequence bucketing
        """

        if self.pre_shuffle_sort:
            random.shuffle(data)
            data = sorted(data, key=len)

        batch_max_length = self.batch_size * self.max_length

        batches = list()
        batches.append(list())

        last_batch_length = 0

        for sample in data:

            current_length = len(sample)

            if last_batch_length + current_length >= batch_max_length:
                batches.append(list())
                last_batch_length = current_length
            else:
                last_batch_length += current_length

            batches[-1].append(sample)

        random.shuffle(batches)

        return batches


class ConversationDataset(Dataset):

    def __init__(self, data: List[List[List[int]]]):
        self.data = data
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
