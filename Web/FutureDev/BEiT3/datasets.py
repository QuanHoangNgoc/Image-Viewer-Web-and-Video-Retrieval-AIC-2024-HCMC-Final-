# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import random
import torch
import glob
from tqdm import tqdm
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

import utils
from glossary import normalize_word
from randaug import RandomAugment


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        split,
        transform,
        tokenizer,
        num_max_bpe_tokens,
        task=None,
    ):
        index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(data_path, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                print(
                    "Load %d image-text pairs from %s. "
                    % (len(items) - offset, index_file)
                )
                offset = len(items)

        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform
        self.split = split

    @staticmethod
    def get_index_files(split):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image_path = os.path.join(self.data_path, image_path)
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[: max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return (
            tokens + [self.pad_token_id] * (max_len - num_tokens),
            padding_mask,
            num_tokens,
        )

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = "{" + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform.transforms:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body


def _write_data_into_jsonl(items, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as writer:
        for data in items:
            writer.write(json.dumps(data, indent=None))
            writer.write("\n")
    print("Write %s with %d items !" % (jsonl_file, len(items)))


def _make_retrieval_coco_karpathy_dataset_index(
    data_path,
    tokenizer,
    split=("train", "restval"),
    split_name="train",
):
    coco_karpathy_split_json_file = os.path.join(data_path, "dataset_coco.json")
    items = []
    image_counter = set()
    print("read %s" % coco_karpathy_split_json_file)
    with open(coco_karpathy_split_json_file, mode="r", encoding="utf-8") as reader:
        data = json.loads(reader.read())
        for item in data["images"]:
            if item["split"] in split:
                image_path = os.path.join(item["filepath"], item["filename"])
                for sent in item["sentences"]:
                    tokens = tokenizer.tokenize(sent["raw"])
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    items.append(
                        {
                            "image_path": image_path,
                            "text_segment": token_ids,
                            "image_id": len(image_counter),
                        }
                    )
                if image_path not in image_counter:
                    image_counter.add(image_path)
    print(
        "Find %d images and %d image-text pairs for karpathy dataset %s split !"
        % (len(image_counter), len(items), split_name)
    )
    index_file = os.path.join(data_path, "coco_retrieval.%s.jsonl" % split_name)
    _write_data_into_jsonl(items, index_file)
    pass


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        transform,
        tokenizer,
        num_max_bpe_tokens,
    ):
        # index_files = self.get_index_files(split, task=task)
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.data_path = data_path
        items = []
        # self.index_files = index_files

        print("Loading data from %s" % data_path)
        # for video in tqdm(sorted(os.listdir(data_path))):
        #     for frame in sorted(os.listdir(os.path.join(data_path, video))):
        #         items.append(
        #             {
        #                 "image_path": os.path.join(data_path, video, frame),
        #                 "text_segment": None,
        #                 "image_id": len(items),
        #             }
        #         )

        for frame in sorted(os.listdir(data_path)):
            items.append(
                {
                    "image_path": os.path.join(data_path, frame),
                    "text_segment": None,
                    "image_id": len(items),
                }
            )
            print(frame, len(items) - 1)

        print("Load %d image-text pairs from %s. " % (len(items), data_path))

        # for _index_file in index_files:
        #     index_file = os.path.join(data_path, _index_file)
        #     with open(index_file, mode="r", encoding="utf-8") as reader:
        #         for line in reader:
        #             data = json.loads(line)
        #             items.append(data)

        #         print("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
        #         offset = len(items)

        self.items = items
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)

    def __getitem__(self, index: int):
        data = dict()
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img
        data["image_id"] = item["image_id"]

        return data

    def __len__(self) -> int:
        return len(self.items)


class RetrievalDataset(BaseDataset):
    @staticmethod
    def get_index_files(split, task=None):
        if split == "train":
            return (f"{task}.train.jsonl",)
        elif split == "val":
            return (f"{task}.val.jsonl",)
        elif split == "test":
            return (f"{task}.test.jsonl",)
        else:
            raise RuntimeError("split %s is not found!" % split)

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        data["image_id"] = self.items[index]["image_id"]
        return data

    @staticmethod
    def make_flickr30k_dataset_index(data_path, tokenizer, karpathy_path):
        with open(os.path.join(karpathy_path, "dataset_flickr30k.json"), "r") as reader:
            captions = json.loads(reader.read())

        captions = captions["images"]
        split2items = defaultdict(list)
        split2images = defaultdict(set)

        for each_item in captions:
            image_path = os.path.join("flickr30k-images", each_item["filename"])
            split = each_item["split"]

            for text_segment in each_item["sentences"]:
                tokens = tokenizer.tokenize(text_segment["raw"])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)

                split2items[split].append(
                    {
                        "image_path": image_path,
                        "text_segment": token_ids,
                        "image_id": len(split2images[split]),
                    }
                )

            assert each_item["filename"] not in split2images[split]
            split2images[split].add(each_item["filename"])

        for split in split2items:
            print(
                "%d images and %d image-text pairs!"
                % (len(split2images[split]), len(split2items[split]))
            )
            _write_data_into_jsonl(
                split2items[split],
                os.path.join(data_path, "flickr30k.%s.jsonl" % split),
            )

    @staticmethod
    def make_coco_dataset_index(data_path, tokenizer):
        _make_retrieval_coco_karpathy_dataset_index(
            data_path, tokenizer, split=("train", "restval"), split_name="train"
        )
        _make_retrieval_coco_karpathy_dataset_index(
            data_path, tokenizer, split=("val",), split_name="val"
        )
        _make_retrieval_coco_karpathy_dataset_index(
            data_path, tokenizer, split=("test",), split_name="test"
        )


def create_dataloader(
    dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False
):
    if is_train or dist_eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        if not is_train and dist_eval and len(dataset) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )

        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(is_train, args):
    if args.task in ["imagenet"]:
        return build_imagenet_transform(is_train, args)

    if is_train:
        t = [
            RandomResizedCropAndInterpolation(
                args.input_size,
                scale=(0.5, 1.0),
                interpolation=args.train_interpolation,
            ),
            transforms.RandomHorizontalFlip(),
        ]
        if args.randaug:
            t.append(
                RandomAugment(
                    2,
                    7,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Equalize",
                        "Brightness",
                        "Sharpness",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                )
            )
        t += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
            ),
        ]
        t = transforms.Compose(t)
    else:
        t = transforms.Compose(
            [
                transforms.Resize((args.input_size, args.input_size), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

    return t


def build_imagenet_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer

    return XLMRobertaTokenizer(args.sentencepiece_model)
