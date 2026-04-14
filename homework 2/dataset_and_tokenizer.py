import torch
import torch.nn as nn
import torch.utils.data as data

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.decoders import BPEDecoder
from transformers import PreTrainedTokenizerFast


class BookDataset(data.Dataset):
    def __init__(self, tokenizer, context_size, book, is_test, test_size=0.1, seed=42):
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.book = book
        self.is_test = is_test
        self.test_size = test_size

        self.dataset = torch.tensor(self.tokenizer.encode(self.book), dtype=torch.long)

        total_length = (self.dataset.size(0) - self.context_size) // self.context_size
        all_indices = torch.randperm(total_length, generator=torch.Generator().manual_seed(seed))

        split = int(total_length * (1 - test_size))
        if is_test:
            self.indices = all_indices[split:]
        else:
            self.indices = all_indices[:split]

        self.length = len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item].item()

        l1 = i * self.context_size
        r1 = (i + 1) * self.context_size
        l2 = i * self.context_size + 1
        r2 = (i + 1) * self.context_size + 1

        return self.dataset[l1:r1], self.dataset[l2:r2]

    def __len__(self):
        return self.length


class MyTokenizerMaker:
    def __init__(self, text, vocab_size, save_dir):
        self.text = text
        self.vocab_size = vocab_size
        self.save_dir = save_dir

        self.__make_tokenizer__()
        self.__make_PyTorch_like_tokenizer__()
        self.__save_tokenzier__()

    def _iter_text_chunks(self, chunk_size=10000):
        for i in range(0, len(self.text), chunk_size):
            chunk = self.text[i:i + chunk_size].strip()
            if chunk:
                yield chunk

    def __make_tokenizer__(self):
        self.tokenizer = Tokenizer(
            BPE(
                unk_token="[UNK]",
                end_of_word_suffix="</w>",
            )
        )
        self.tokenizer.normalizer = Lowercase()
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = BPEDecoder(suffix="</w>")

        special_tokens = [
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]",
        ]

        self.trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True,
            end_of_word_suffix="</w>",
        )

        self.tokenizer.train_from_iterator(
            self._iter_text_chunks(),
            trainer=self.trainer,
        )

    def __make_PyTorch_like_tokenizer__(self):
        self.hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )

    def __save_tokenzier__(self):
        self.hf_tokenizer.save_pretrained(f"{self.save_dir}/tokenizer")

    def get_tokenizer(self):
        return self.hf_tokenizer