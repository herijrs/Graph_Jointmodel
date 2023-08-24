import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

import utils
from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):  # 自定义输出类，可以控制输出样本的格式-json

    """
        A single training/test example for simple sequence classification. 一个单独的样本实例
        一个样本完全可以用一个dict来表示，但使用InputExample类，作为一个python类，具有一些方便之处
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            intent_label: (Optional) string. The intent label of the example.
            slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid  # 每个样本的独特序号
        self.words = words  # 样本的输入序列
        self.intent_label = intent_label  # 样本的intent标签
        self.slot_labels = slot_labels  # 样本的slot标签序列

    def __repr__(self):
        # 默认为：“类名 + object at + 内存地址” 这样的信息表示这个实例
        # 重写需要输出的信息
        # print(input_example) 时显示
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        # __dict__：
        # 类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的
        # 对象实例的__dict__中存储一些self.xxx的东西
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):  # 数据处理器类
    """Processor for the JointBERT data set """
    # JointBert项目的数据处理器
    def __init__(self, args):
        self.args = args  # 项目的参数配置
        # 加载处理好的意图标签和槽位标签
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        # 每个数据集的文件夹中数据格式一致，文件名格式也一致
        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'

    # 执行读取文件的函数
    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        # 以行为单位进行读取
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    # 对每一个样本进行处理
    def _create_examples(self, texts, intents, slots, set_type):
        """
        Creates examples for the training and dev sets.
            Args:
        texts: list. Sequence of unsplitted texts.需要处理的文本组成的列表
        intents: list. Sequence of intent labels. 意图label组成的列表
        slots: list. Sequence of unsplitted slot labels. 槽位label组成的列表
        set_type: str. train\ dev\ test 训练集、验证集、测试集

        """
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            # 如果验证集或测试集中的标签不在训练集中，将其标为UNK
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                # 如果验证集或测试集中的标签不在训练集中，将其标为UNK
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            # 进行验证 防止由于标签遗漏导致的错误
            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
            判断输入的是 训练集、验证集还是测试集，按照对应的路径读取文件
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)


processors = {
    "atis": JointProcessor,
    "snips": JointProcessor
}

# 将数据处理成Bert能够理解的特征
def convert_examples_to_features(examples, # 输入的训练样本
                                 max_seq_len, # 样本最大长度
                                 tokenizer, # subword tokenizer
                                 pad_token_label_id=-100, # 新加入的标签编号
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):

    """
    将之前读取的数据进行添加[CLS][SEP]标记，padding操作
    Args:
        examples: 样本实例列表
        max_seq_len: 最大长度
        tokenizer:
        pad_token_label_id:
        cls_token_segment_id: 取0
        pad_token_segment_id: 取0
        sequence_a_segment_id: 取0
        mask_padding_with_zero: attention mask
    Returns:
    """
    # Setting based on the current model type
    cls_token = tokenizer.cls_token  # [CLS]
    sep_token = tokenizer.sep_token  # [SEP]
    unk_token = tokenizer.unk_token  # [UNK]
    pad_token_id = tokenizer.pad_token_id  # [PAD]编号为0

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        # bert采用的tokenizer可能会把一个单词分成多个subword，将第一个subword标记为slot label,其他标记为pad label
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word 不能识别的word标记为UNK
                '''
                {
                    '0' : 0,
                    'B-ENT' : 1 ,
                    'I-ENT' : 2
                }
                '''
                # 例如 principle:prin cip le
                # B-ENT:B-ENT,X,X: 1,-100,-100 新添标签 X 新的label类 (最常见)
                # B-ENT:B-ENT,I-ENT,I-ENT: 1,2,2 实体未结束的label类 I-ENT
                # B-ENT:B-ENT,0,0: 1,0,0 非实体 0
                # B-ENT:B-ENT,B-ENT,B-ENT,: 1,1,1 实体的开头部分

            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        # 若句子太长将其截断
        # 为保证 tokens 和 slot_labels 两者长度一致，需要对slot_labels做相同操作
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # 将单词转化为ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    return dataset