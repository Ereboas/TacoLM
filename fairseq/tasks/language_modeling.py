# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
    FairseqDataset
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


class ModifiedLMDataset(FairseqDataset):

    def __init__(
        self,
        manifest_path,
        pad,
        eos,
        bos,
        is_ar,
        preload,
        RVQLayer = 8,
        left_pad_text = True,
        use_bos_to_sep = False,
    ):
        #TODO: 文件名可以用 fairseq/data/audio/raw_audio_dataset.py 中 FileAudioDataset 的实现方式, 进行压缩
        #? 23/03/18 
        self.text_compressor = TextCompressor(level=TextCompressionLevel.none)
        self.fname_prefixs = []
        self.sizes = []
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                fname, _size = line.strip().split("\t")
                dot_index = fname.rfind('.')
                self.fname_prefixs.append(self.text_compressor.compress(fname[ :dot_index]))
                self.sizes.append(int(_size))

        # ! TODO: 规避掉fname_prefixs数组的Memory Problem (when not preloaded).
        import pyarrow
        self.fname_prefixs = pyarrow.array(self.fname_prefixs)

        self.sizes = np.array(self.sizes, dtype=np.int64)
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.is_ar = is_ar
        self.preload = preload
        self.RVQLayer = RVQLayer

        self.left_pad_text = left_pad_text
        self.use_bos_to_sep = use_bos_to_sep

        if self.preload:
            self._preload_dataset()
    
    def _preload_dataset(self):
        def _get_indices(length_list):
            lengths = torch.LongTensor(length_list)
            cumsum = torch.cumsum(lengths, dim=0)
            s = torch.cat([torch.LongTensor([0]), cumsum[:-1]])
            indices = torch.stack([s, cumsum]).t_()
            return indices

        def _get_data(suffix='.bpe'):
            from tqdm import tqdm
            codes, lengths = [], []
            logger.info(f"_preload_dataset: {suffix}")
            for i, fname_prefix in enumerate(tqdm(self.fname_prefixs)):
                fname_prefix = fname_prefix.as_py()
                fname_prefix = self.text_compressor.decompress(fname_prefix)
                path = os.path.join(self.root_dir, fname_prefix + suffix)
                code = torch.load(path)
                lengths.append(code.size(-1))
                codes.append(code)
            indices = _get_indices(lengths)
            codes = torch.cat(codes, dim=-1)
            return indices, codes
        self.bpe_indices, self.bpe_codes = _get_data('.bpe')
        self.qnt_indices, self.qnt_codes = _get_data('.qnt')
    
    def __getitem__(self, index):
        if self.preload:
            bpe_s, bpe_e = self.bpe_indices[index]
            qnt_s, qnt_e = self.qnt_indices[index]
            bpe_code = self.bpe_codes[bpe_s: bpe_e]
            qnt_code = self.qnt_codes[:, qnt_s: qnt_e]
        else:
            fname_prefix = self.fname_prefixs[index]
            fname_prefix = fname_prefix.as_py()
            fname_prefix = self.text_compressor.decompress(fname_prefix)
            bpe_path = os.path.join(self.root_dir, fname_prefix +'.bpe')
            qnt_path = os.path.join(self.root_dir, fname_prefix +'.qnt')
            bpe_code = torch.load(bpe_path)
            qnt_code = torch.load(qnt_path)
        
        bpe_code = bpe_code + 4 + 1024
        qnt_code = qnt_code + 4
        '''
        eos_column_code = torch.LongTensor([self.eos]).repeat(self.RVQLayer, 1)
        bpe_column_code = bpe_code.repeat(self.RVQLayer, 1)
        data_source = torch.cat([
            bpe_column_code, eos_column_code, qnt_code, eos_column_code
        ], dim=1)

        ar_source = data_source[0]
        ar_target = torch.cat([ar_source[1:], torch.LongTensor([self.eos])])
        #to-do NAR部分还是有些问题
        separator = len(bpe_code) + 1 # * From separator, the qnt codes begin. 
        return {"id": index, "separator": separator, "ar_source": ar_source, "ar_target": ar_target, "data_source": data_source}
        '''
        bpe_code = torch.cat([
            torch.LongTensor([self.bos]),
            bpe_code, 
            torch.LongTensor([self.bos if self.use_bos_to_sep else self.eos])
        ])
        qnt_code = torch.cat([
            torch.LongTensor([self.bos]).repeat(self.RVQLayer, 1),
            qnt_code, 
            torch.LongTensor([self.eos]).repeat(self.RVQLayer, 2)
        ], dim=1)
        return {"id": index, "bpe_code_with_sep": bpe_code, "qnt_code_with_double_eos": qnt_code}
        
    def __len__(self):
        return len(self.fname_prefixs)
    
    def size(self, index):
        return self.sizes[index]

    def num_tokens(self, index):
        return self.sizes[index]

    def collater(self, samples):
        return_dict = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "nsentences": len(samples),
        }

        if self.is_ar:
            train_nar_layer = None
            bpe_codes_with_pad_and_sep = data_utils.collate_tokens(
                [s["bpe_code_with_sep"] for s in samples],
                self.pad,
                left_pad=self.left_pad_text,
            )
            qnt_codes_with_pad_and_double_eos = data_utils.collate_tokens(
                [s["qnt_code_with_double_eos"][0] for s in samples],
                self.pad,
            )
            #@params (batch_size, len)
            ar_source = torch.cat([
                bpe_codes_with_pad_and_sep, qnt_codes_with_pad_and_double_eos
            ], dim=1)
            src_tokens = ar_source[:, :-1]
            target_tokens = ar_source[:, 1:]
            ntokens = sum(
                s["bpe_code_with_sep"].size(-1) + s["qnt_code_with_double_eos"].size(-1) - 1
                for s in samples
            )
            src_lengths = torch.LongTensor([
                s["bpe_code_with_sep"].size(-1) + s["qnt_code_with_double_eos"].size(-1) - 1 
                for s in samples
            ])
        else:
            train_nar_layer = torch.randint(1, self.RVQLayer, (1,)).item() #* [1, RVQLayer-1)
            # * 从 1 ~ RVQLayer-1 中随机选取 1 个整数
            bpe_codes_with_pad_and_sep = data_utils.collate_tokens(
                [s["bpe_code_with_sep"] for s in samples],
                self.pad,
                left_pad=self.left_pad_text,
            )
            qnt_codes_with_pad_and_double_eos = [
                data_utils.collate_tokens(
                    [s["qnt_code_with_double_eos"][i] for s in samples],
                    self.pad,
                )
                for i in range(train_nar_layer)
            ]
            nar_source = [
                torch.cat([
                    bpe_codes_with_pad_and_sep, qnt_codes_with_pad_and_double_eos[i]
                ], dim=1)
                for i in range(train_nar_layer)
            ]
            src_tokens = [
                nar_source[i][:, :-1]
                for i in range(train_nar_layer)
            ]
            target_tokens = torch.cat([
                bpe_codes_with_pad_and_sep,
                data_utils.collate_tokens(
                    [s["qnt_code_with_double_eos"][train_nar_layer] for s in samples],
                    self.pad,
                )
            ], dim=1)[:, 1:]

            ntokens = sum(
                s["bpe_code_with_sep"].size(-1) + s["qnt_code_with_double_eos"].size(-1) - 1
                for s in samples
            )
            src_lengths = torch.LongTensor([
                s["bpe_code_with_sep"].size(-1) + s["qnt_code_with_double_eos"].size(-1) - 1 
                for s in samples
            ])

        return_dict |= {
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "bpe_length": bpe_codes_with_pad_and_sep.size(1),
                "train_nar_layer": train_nar_layer,
            },
            "target": target_tokens,
        }

        return return_dict

@dataclass
class LanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    modified: bool = field(
        default=False, metadata={"help": "Set to True to apply the modified LM."}
    )
    # NOTE: 变量名为 ar_task, 命令行用 --ar-task
    ar_task: bool = field(
        default=False, metadata={"help": "Set to True to apply the modified Ar LM."}
    )
    preload_dataset: bool = field(
        default=False, metadata={"help": "Preload Qnt & Bpe files in memory."}
    )
    rvq_layers: int = field(
        default=8,
        metadata={"help": "Layers of RVQ"},
    )
    left_pad_text: bool = field(
        default=False, metadata={"help": "Left padding for text."}
    )
    use_bos_to_sep: bool = field(
        default=False, metadata={"help": "Use <bos> to separate text and audio; if set to False, use <eos> by default."}
    )
    valid_block: Optional[str] = field(
        default="size:10000",
        metadata={"help": "either size:value or splits:value, " 
                          "the former is the block size, the latter is the number of blocks"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("language_modeling", dataclass=LanguageModelingConfig)
class LanguageModelingTask(LegacyFairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary

        self.chunk_size = args.decoder_chunk_size if hasattr(args, 'decoder_chunk_size') else -1
        self.is_mega_lm = True if hasattr(args, 'decoder_chunk_size') else False

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        #TODO: 直接设置Dictionary.
        dictionary = None
        output_dictionary = None
        if args.modified:
            dictionary = Dictionary()
            for i in range(2000 + 1024):
                dictionary.add_symbol(i)  #NOTE: 这里symbol是INT而不是STR
            output_dictionary = dictionary
            logger.info("dictionary: 2000 + 1024 types")
            return (dictionary, output_dictionary)

        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        
        # NOTELINE: TODO:
        # to-do: 读取方式改变为mmap
        # NOTELINE: 23/02/27

        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        if self.args.modified:
            manifest_path = os.path.join(self.args.data, f"{split}.tsv")
            self.datasets[split] = ModifiedLMDataset(
                manifest_path=manifest_path,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                bos=self.dictionary.bos(),
                is_ar = self.args.ar_task,
                preload = self.args.preload_dataset,
                RVQLayer = self.args.rvq_layers,
                left_pad_text=self.args.left_pad_text,
                use_bos_to_sep=self.args.use_bos_to_sep,
            )
            return

        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # each process has its own copy of the raw data (likely to be an np.memmap)
        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = (
                self.args.batch_size_valid if "valid" in split else self.args.batch_size
            )

        self.datasets[split] = MonolingualDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
