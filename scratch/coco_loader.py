import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

data_path = "D:/ml_research/datasets/coco/val2017"
ann_path = "D:/ml_research/datasets/coco/annotations/captions_val2017.json"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def main():     # wrapped so it doesn't run again when worker processes are spawned rather than forked on Windows/MacOS
    coco_val = datasets.CocoDetection(root=data_path, annFile=ann_path, transform=transform)
    train_loader = DataLoader(
        coco_val,
        batch_size=8,
        shuffle=True,
    )

    img, target = next(iter(train_loader))
    print(f"Success: Image batch shape: {img.shape}")   # should be [8, 3, 224, 224]


if __name__ == '__main__':
    main()