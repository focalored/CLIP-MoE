import torch
import random
from torchvision import datasets, transforms
import torch.utils.data as data
import clip

"""
DATA STRUCTURE: COCO Captions
Batch size (B): 8
Captions per image (C): 5

Input (x): Tensor[float32] (B, 3, 224, 224)     -> RGB pixel values
Target (y): List[dict] (C,)                     -> List of C sets of captioned batches
[
    {
        'image_id': Tensor[int64] (B,),         -> global image identifiers
        'id':       Tensor[int64] (B,),         -> global caption identifiers
        'caption':  List[str] (B,)              -> raw strings for captions (need tokenization)
    },
    ... (C repeats)
]

- Image and caption ids (JSON numbers) are stored in int64 tensors (contiguous multi-dimensional array) by default
- Captions (JSON strings) are stored in lists because they have different lengths and can't be stacked in continguous memory
"""

data_path = "/project/osprey/scratch/liv/datasets/coco/val2017"
ann_path = "/project/osprey/scratch/liv/datasets/coco/annotations/captions_val2017.json"


class CocoForClipMoE(datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super(CocoForClipMoE, self).__init__(root, annFile, transform)
        self.output_idx = False
    
    def __getitem__(self, index):
        image, target = super(CocoForClipMoE, self).__getitem__(index)

        caption = ""
        if len(target) > 0:
            # pick one random caption out of five
            chosen_ann = random.choice(target)
            caption = chosen_ann['caption']
        caption_short = caption.split(". ")[0]

        if self.output_idx:
            return image, caption, index
        return image, caption, caption_short


def clip_collate_fn(batch):
    """
    :param batch: batch is a list of tuples:
    [
        (img1, cap1, short1),
        (img2, cap2, short2),
        ...
    ]

    zip(*batch) transposes list of rows into tuples, one for each column:
    (img1, img2, ...),
    (cap1, cap2, ...),
    (short1, short2, ...)
    """
    images, captions, captions_short = zip(*batch)
    images = torch.stack(images, 0) # stack into a (B, 3, 224, 224) tensor
    return images, list(captions), list(captions_short)


def main():
    # wrapped in main() so this doesn't run again when worker processes are spawned rather than forked on Windows/MacOS
    _, preprocess = clip.load("ViT-L/14")
    coco_val = CocoForClipMoE(root=data_path, annFile=ann_path, transform=preprocess)
    train_loader = data.DataLoader(
        coco_val,
        batch_size=8,
        shuffle=True,
        collate_fn=clip_collate_fn,
        num_workers=2,
        pin_memory=True,     # for transfering tensors to GPU
    )

    imgs, caps, caps_short = next(iter(train_loader))  # next batch

    imgs = imgs.to("cpu")
    tokens = clip.tokenize(caps).to("cpu")
    tokens_short = clip.tokenize(caps_short).to("cpu")

    print(f"Image batch shape: {imgs.shape}")
    print(f"Image tensor mean: {imgs.mean():.4f}")   # should be near 0 due to normalization
    print(f"Tokens shape: {tokens.shape}")  # should be (B, 77) after CLIP tokenization
    print(f"Tokens device: {tokens.device}")
    print(f"\nShort captions:")
    for i in range(len(caps_short)):
        print(f"{i+1}. {caps_short[i]}")

if __name__ == '__main__':
    main()
