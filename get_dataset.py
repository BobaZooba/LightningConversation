import copy
import gzip
import logging
import os
import pickle
import re
import shutil
from argparse import ArgumentParser, Namespace
from typing import List, Dict, Tuple, Iterable, Generator, Optional, Any

import requests
import youtokentome as yttm
from numpy import random
from tqdm import tqdm
import json

from src.tokenizer import Tokenizer


class BaseCollector:

    def __init__(self, config: Namespace):

        self.config = config
        self.data_dir = os.path.join(os.getcwd(), self.config.data_dir)

        self.bpe_model_path = os.path.join(self.data_dir, 'bpe.model')

        self.bpe_train_path = os.path.join(self.data_dir, 'bpe_train.txt')

        self.sep_token = f' {self.config.sep_token} '
        if self.config.context_token is None:
            self.context_token = None
        else:
            self.context_token = f' {self.config.context_token} '

        self.train_dir = os.path.join(self.data_dir, 'train')
        self.validation_dir = os.path.join(self.data_dir, 'validation')
        # validation size is too small to split to chunks
        self.validation_file_path = os.path.join(self.validation_dir, 'chunk_0.pkl')

        self.tokenizer = None
        self.data_generator = None

        self.validation_data = list()

        self.total_bpe_samples = None
        self.total_collecting_samples = None

    @staticmethod
    def download_file(url: str, save_path: str, verbose: bool = False):
        try:
            filename = save_path.split('/')[-1]
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=8192),
                                      desc=f'Download {filename}',
                                      disable=not verbose):
                        if chunk:
                            f.write(chunk)
        except KeyboardInterrupt as exception:
            os.remove(save_path)
            raise exception
        except Exception as exception:
            os.remove(save_path)
            raise exception

    @staticmethod
    def read_file(path: str) -> Iterable[bytes]:
        file_object = gzip.open(path, mode='r')
        for sample in file_object:
            yield sample

    @staticmethod
    def make_dir(path: str, override: bool = False):
        if override:
            shutil.rmtree(path, ignore_errors=True)

        try:
            os.mkdir(path)
        except FileExistsError:
            pass

    @staticmethod
    def pickling(python_object: Any, path: str):
        with open(path, mode='wb') as file_object:
            pickle.dump(python_object, file_object)

    def filter_rules(self, text1: str, text2: Optional[str] = None) -> bool:

        len_text1 = len(text1)

        if text2 is None:
            return self.config.min_chars <= len_text1 <= self.config.max_chars

        len_text2 = len(text2)

        if text1 == text2 \
                or min(len_text1, len_text2) < self.config.min_chars \
                or max(len_text1, len_text2) > self.config.max_chars \
                or sum([len_text1, len_text2]) > self.config.max_chars:
            return False
        else:
            return True

    def tokenized_filter_rules(self, tokenized_text: List[int]) -> bool:

        current_sep_index = tokenized_text.index(self.tokenizer.sep_index)

        len_unknowns = tokenized_text.count(self.tokenizer.unk_index)
        len_query = current_sep_index - 1
        len_response = len(tokenized_text) - current_sep_index - 2

        if not (self.config.min_tokens < len(tokenized_text) < self.config.max_tokens) \
                or len_query < self.config.min_tokens_query \
                or len_response < self.config.min_tokens_response \
                or len_unknowns > self.config.max_unknowns:
            return False
        else:
            return True

    @staticmethod
    def text_processing(text: str) -> str:
        text = re.sub('[\ud83d\ude0a\ude03]', '', text)
        text = re.sub('[\\[(](.*?)[\\])]', '', text)

        text = text.strip(' -\n\r,.')

        return text

    def to_dict(self) -> Dict:
        return copy.deepcopy(self.__dict__)

    def save_vocab(self):
        bpe = yttm.BPE(model=self.bpe_model_path)
        vocab = bpe.vocab()

        with open(os.path.join(self.config.data_dir, 'vocab.txt'), mode='w') as file_object:
            file_object.write('\n'.join(vocab))

    def collect_chunk(self, chunk: List[List[int]], n_chunk: int):
        tokenized_chunk = self.tokenizer.tokenize(chunk)
        tokenized_chunk = [sample for sample in tokenized_chunk if self.tokenized_filter_rules(sample)]

        random.shuffle(tokenized_chunk)
        current_chunk = sorted(tokenized_chunk, key=len)

        chunk_path = os.path.join(self.train_dir, f'chunk_{n_chunk}.pkl')
        self.pickling(python_object=current_chunk, path=chunk_path)

    def generate_data(self) -> Generator[Any, None, None]:
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def train_bpe(self):

        bpe_train_file = open(self.bpe_train_path, mode='w')
        saved_bpe_samples = 0

        progress_bar = tqdm(total=self.total_bpe_samples,
                            desc='Loading for BPE',
                            disable=not self.config.verbose)

        for samples in self.generate_data():

            if saved_bpe_samples >= self.config.n_bpe_train_samples:
                break

            if not samples:
                continue

            bpe_train_file.write('\n'.join(samples) + '\n')
            saved_bpe_samples += len(samples)
            progress_bar.update(n=len(samples))

        progress_bar.close()
        bpe_train_file.close()

        yttm.BPE.train(data=self.bpe_train_path,
                       vocab_size=self.config.vocab_size,
                       model=self.bpe_model_path,
                       coverage=self.config.bpe_coverage)

        self.save_vocab()

    def collect(self):

        train_saved_samples = 0
        current_n_chunk = 0
        chunk = list()

        progress_bar = tqdm(total=self.total_collecting_samples,
                            desc='Collecting dataset',
                            disable=not self.config.verbose)

        for samples in self.generate_data():

            if train_saved_samples >= self.config.max_train_samples:
                break

            if not samples:
                continue

            if len(self.validation_data) < self.config.min_validation_size \
                    and random.random() < self.config.validation_prob:
                self.validation_data.extend(samples)
            else:
                chunk.extend(samples)

                if len(chunk) >= self.config.chunk_size:
                    self.collect_chunk(chunk=chunk, n_chunk=current_n_chunk)
                    current_n_chunk += 1
                    train_saved_samples += len(chunk)
                    chunk.clear()

                progress_bar.update(n=len(samples))

        progress_bar.clear()

        if self.validation_data:
            random.shuffle(self.validation_data)
            self.validation_data = sorted(self.validation_data, key=len)
            tokenized_validation_data = self.tokenizer.tokenize(self.validation_data)
            tokenized_validation_data = [sample for sample in tokenized_validation_data
                                         if self.tokenized_filter_rules(sample)]
            random.shuffle(tokenized_validation_data)
            self.pickling(python_object=tokenized_validation_data, path=self.validation_file_path)

        if chunk:
            self.collect_chunk(chunk=chunk, n_chunk=current_n_chunk)
            current_n_chunk += 1
            train_saved_samples += len(chunk)
            chunk.clear()

    def run(self):

        if self.config.download:
            logger.info('Download')
            collector.download()

        if self.config.train_bpe:
            logger.info('Train BPE')
            collector.train_bpe()

        self.tokenizer = Tokenizer(tokenizer_path=self.bpe_model_path,
                                   need_bos=True,
                                   need_eos=True,
                                   sep_token=self.config.sep_token,
                                   context_token=self.config.context_token)

        if self.config.collect_data:
            logger.info('Parse data')
            self.make_dir(self.train_dir, override=True)
            self.make_dir(self.validation_dir, override=True)
            collector.collect()


class AmazonCollector(BaseCollector):
    """
    Dataset from https://github.com/PolyAI-LDN/conversational-datasets/tree/master/amazon_qa
    """

    URLS = {
        'single': [
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Appliances.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Arts_Crafts_and_Sewing.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Automotive.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Baby.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Beauty.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Cell_Phones_and_Accessories.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Clothing_Shoes_and_Jewelry.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Electronics.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Grocery_and_Gourmet_Food.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Health_and_Personal_Care.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Home_and_Kitchen.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Industrial_and_Scientific.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Musical_Instruments.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Office_Products.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Patio_Lawn_and_Garden.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Pet_Supplies.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Software.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Sports_and_Outdoors.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Tools_and_Home_Improvement.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Toys_and_Games.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/qa_Video_Games.json.gz'
        ],
        'multiple': [
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Automotive.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Baby.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Beauty.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Cell_Phones_and_Accessories.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Clothing_Shoes_and_Jewelry.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Electronics.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Grocery_and_Gourmet_Food.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Health_and_Personal_Care.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Home_and_Kitchen.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Musical_Instruments.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Office_Products.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Patio_Lawn_and_Garden.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Pet_Supplies.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Sports_and_Outdoors.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Tools_and_Home_Improvement.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Toys_and_Games.json.gz',
            'http://jmcauley.ucsd.edu/data/amazon/qa/icdm/QA_Video_Games.json.gz'
        ]
    }

    def __init__(self, config: Namespace):
        super().__init__(config=config)

        self.single_dir = os.path.join(self.data_dir, 'single')
        self.multiple_dir = os.path.join(self.data_dir, 'multiple')

    def create_sample(self, text1: str, text2: str) -> str:
        return self.text_processing(text1) + self.sep_token + self.text_processing(text2)

    def parse_single_file_sample(self, sample: Dict) -> List[str]:
        if self.filter_rules(sample['question'], sample['answer']):
            return [self.create_sample(text1=sample['question'], text2=sample['answer'])]
        else:
            return []

    def parse_multiple_file_sample(self, sample: Dict) -> List[str]:
        output_data = list()

        questions = sample['questions']
        for que in questions:
            que_text = que['questionText']
            for answer in que['answers']:
                ans_text = answer['answerText']
                if self.filter_rules(que_text, ans_text):
                    output_data.append(self.create_sample(text1=que_text,
                                                          text2=ans_text))

        return output_data

    def get_paths(self) -> List[Tuple[str, str]]:
        paths = list()

        for file_name in os.listdir(self.single_dir):
            if file_name.startswith('qa') and file_name.endswith('.gz'):
                paths.append((os.path.join(self.single_dir, file_name), 'single'))

        for file_name in os.listdir(self.multiple_dir):
            if file_name.startswith('QA') and file_name.endswith('.gz'):
                paths.append((os.path.join(self.multiple_dir, file_name), 'multiple'))

        random.shuffle(paths)

        return paths

    def generate_data(self) -> Generator[Any, None, None]:
        paths = self.get_paths()

        for path, file_type in paths:
            for qa_sample in self.read_file(path):
                qa_sample = eval(qa_sample)
                if file_type == 'single':
                    parsed_samples = self.parse_single_file_sample(qa_sample)
                elif file_type == 'multiple':
                    parsed_samples = self.parse_multiple_file_sample(qa_sample)
                else:
                    raise ValueError('Not expected file_type')
                yield parsed_samples

    def train_bpe(self):

        paths = self.get_paths()

        bpe_train_file = open(self.bpe_train_path, mode='w')
        saved_bpe_samples = 0

        progress_bar = tqdm(total=self.total_bpe_samples,
                            desc='Loading for BPE',
                            disable=not self.config.verbose)

        for file_path, file_type in paths:

            if saved_bpe_samples >= self.config.n_bpe_train_samples:
                break

            samples = list()

            for qa_sample in self.read_file(file_path):
                qa_sample = eval(qa_sample)
                if file_type == 'single':
                    samples.append(qa_sample['question'])
                    samples.append(qa_sample['answer'])
                elif file_type == 'multiple':
                    questions = qa_sample['questions']
                    for que in questions:
                        samples.append(que['questionText'])
                        for answer in que['answers']:
                            samples.append(answer['answerText'])
                else:
                    raise ValueError('Not expected file_type')

            samples = [self.text_processing(text) + ' ' + self.config.sep_token for text in samples]

            if not samples:
                continue

            bpe_train_file.write('\n'.join(samples) + '\n')
            saved_bpe_samples += len(samples)
            progress_bar.update(n=len(samples))

        progress_bar.close()
        bpe_train_file.close()

        yttm.BPE.train(data=self.bpe_train_path,
                       vocab_size=self.config.vocab_size,
                       model=self.bpe_model_path,
                       coverage=self.config.bpe_coverage)

        self.save_vocab()

    def download(self):

        self.make_dir(self.data_dir)

        for key in self.URLS:

            current_dir = os.path.join(self.data_dir, key)

            self.make_dir(current_dir)

            for url in self.URLS[key]:

                file_name = url.split('/')[-1]
                file_path = os.path.join(current_dir, file_name)

                if not os.path.isfile(file_path):
                    self.download_file(url=url, save_path=file_path, verbose=self.config.verbose)
                    logger.info('File %s downloaded', file_path)
                else:
                    logger.info('File %s exist', file_path)


class OpenSubtitlesCollector(BaseCollector):
    """
    Dataset from https://github.com/PolyAI-LDN/conversational-datasets/tree/master/opensubtitles
    """
    URL = 'http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.en.gz'
    FILE_NAME = 'en.txt.gz'

    def __init__(self, config: Namespace):

        super().__init__(config=config)

        self.input_file_path = os.path.join(self.data_dir, self.FILE_NAME)
        if not os.path.isfile(self.input_file_path):
            self.input_file_path = None

        self.total_bpe_samples = self.config.n_bpe_train_samples
        self.total_collecting_samples = self.config.max_train_samples

    def create_samples(self, phrase: str, context: List[str]) -> List[str]:
        samples = list()

        for n in range(len(context)):
            context_query = self.context_token.join(context[-n:])

            if self.filter_rules(context_query, phrase):
                samples.append(context_query + self.sep_token + phrase)

        return samples

    @staticmethod
    def text_processing(text: str) -> str:
        text = re.sub('(?:^|(?:[.!?]\\s))(\\w+):', '', text)
        text = re.sub('[\ud83d\ude0a\ude03]', '', text)
        text = re.sub('[\\[(](.*?)[\\])]', '', text)

        text = text.strip(' -\n\r,.')

        return text

    def generate_data(self) -> Generator[Any, None, None]:
        n_empty_lines = 0

        context = list()

        for line in self.read_file(self.input_file_path):
            line = line.decode('utf-8').strip()

            if not line:
                n_empty_lines += 1
            else:
                n_empty_lines = 0

            if n_empty_lines >= 1000:
                break

            phrase = self.text_processing(line)

            if not self.filter_rules(phrase):
                context.clear()
                continue

            if not context:
                context.append(phrase)
                continue

            context = context[-self.config.max_n_context:]
            samples = self.create_samples(phrase, context)
            yield samples
            context.append(phrase)

    def download(self):
        if self.input_file_path is None:
            self.input_file_path = os.path.join(self.data_dir, self.FILE_NAME)
            self.download_file(url=self.URL,
                               save_path=self.input_file_path,
                               verbose=self.config.verbose)
            logger.info('File %s downloaded', self.input_file_path)
        else:
            logger.info('File %s exist', self.input_file_path)

    def train_bpe(self):

        bpe_train_file = open(self.bpe_train_path, mode='w')
        saved_bpe_samples = 0

        with tqdm(desc='Loading for BPE',
                  total=self.config.n_bpe_train_samples,
                  disable=not self.config.verbose) as progress_bar:
            for line in self.read_file(self.input_file_path):

                phrase = self.text_processing(line.decode('utf-8'))

                if self.filter_rules(phrase):

                    phrase += self.sep_token

                    if self.context_token is not None:
                        phrase += self.context_token

                    bpe_train_file.write(phrase + '\n')
                    saved_bpe_samples += 1
                    progress_bar.update()

                if saved_bpe_samples >= self.config.n_bpe_train_samples:
                    break

        bpe_train_file.close()

        yttm.BPE.train(data=self.bpe_train_path,
                       vocab_size=self.config.vocab_size,
                       model=self.bpe_model_path,
                       coverage=self.config.bpe_coverage)

        self.save_vocab()


if __name__ == '__main__':

    logger = logging.getLogger(__file__)

    parser = ArgumentParser()

    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--sep_token', type=str, default='<SEP>')
    parser.add_argument('--context_token', type=str, default=None)
    parser.add_argument('--max_n_context', type=int, default=3)
    parser.add_argument('--max_train_samples', type=int, default=int(1.e+9))
    parser.add_argument('--n_bpe_train_samples', type=int, default=int(5.e+7))

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--train_bpe', action='store_true')
    parser.add_argument('--collect_data', action='store_true')

    parser.add_argument('--chunk_size', type=int, default=int(1.5e+6))
    parser.add_argument('--min_validation_size', type=int, default=100000)
    parser.add_argument('--validation_prob', type=float, default=0.1)

    parser.add_argument('--min_chars', type=int, default=25)
    parser.add_argument('--max_chars', type=int, default=512)
    parser.add_argument('--min_tokens', type=int, default=10)
    parser.add_argument('--min_tokens_query', type=int, default=3)
    parser.add_argument('--min_tokens_response', type=int, default=3)
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--max_unknowns', type=int, default=3)

    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--bpe_coverage', type=float, default=0.999)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.data_source == 'amazon':
        collector = AmazonCollector(args)
    elif args.data_source == 'opensubtitles':
        collector = OpenSubtitlesCollector(args)
    else:
        collector = BooksCollector(args)

    collector.run()
