"""Generic image-text pair data module backed by HuggingFace datasets.

Handles tokenization and image preprocessing via the backbone's ``AutoProcessor``.
"""

from __future__ import annotations

from typing import Optional

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor


class ImageTextDataset:
    """Wraps a HuggingFace dataset split, applying processor on-the-fly."""

    def __init__(self, hf_dataset, processor, image_column, caption_column, max_length):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_column = image_column
        self.caption_column = caption_column
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        caption = item[self.caption_column]
        if isinstance(caption, list):
            caption = caption[0]

        if image.mode != "RGB":
            image = image.convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


class ImageTextDataModule(L.LightningDataModule):
    """LightningDataModule that loads image-text pairs from HuggingFace Hub or local disk.

    Config parameters map directly to ``datasets.load_dataset`` and
    ``transformers.AutoProcessor``.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        dataset_name: str = "nlphuji/flickr30k",
        dataset_config: Optional[str] = None,
        image_column: str = "image",
        caption_column: str = "caption",
        train_split: str = "test",
        val_split: str = "test",
        test_split: str = "test",
        max_length: int = 77,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.image_column = image_column
        self.caption_column = caption_column
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.processor = AutoProcessor.from_pretrained(model_name)

    def setup(self, stage: Optional[str] = None):
        load_kw = {}
        if self.dataset_config:
            load_kw["name"] = self.dataset_config

        if stage in (None, "fit"):
            raw = load_dataset(self.dataset_name, split=self.train_split, **load_kw)
            self.train_dataset = ImageTextDataset(
                raw, self.processor, self.image_column, self.caption_column, self.max_length
            )
            raw_val = load_dataset(self.dataset_name, split=self.val_split, **load_kw)
            self.val_dataset = ImageTextDataset(
                raw_val, self.processor, self.image_column, self.caption_column, self.max_length
            )
        if stage in (None, "test"):
            raw_test = load_dataset(self.dataset_name, split=self.test_split, **load_kw)
            self.test_dataset = ImageTextDataset(
                raw_test, self.processor, self.image_column, self.caption_column, self.max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
