"""
A pre-trained Mask R-CNN model for RTS mapping
Original codes from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Modified by Yiwen lin

Dataset setting
"""

import os
import numpy as np
import torch
from PIL import Image
from utilities.image_augmentation import augment_seq

# For RTS training images
class PennFudanDataset_train_class1(torch.utils.data.Subset):
    def __init__(self, root, transforms=None, augment_list=[]):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        img_dir = os.path.join(root, "split_256_nrg/split_images_train_class1")
        img_list = []
        for file in os.listdir(img_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                img_list.append(file)
        img_list = list(sorted(img_list))

        img_list_array = []
        for img_name in img_list:
            img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
            img_array = np.array(img)
            img_list_array.append(img_array)

        mask_dir = os.path.join(root, "split_256_nrg/split_labels_train_class1")
        mask_list = []
        for file in os.listdir(mask_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                mask_list.append(file)
        mask_list = list(sorted(mask_list))

        mask_list_array = []
        for mask_name in mask_list:
            mask = Image.open(os.path.join(mask_dir, mask_name))
            mask_array = np.array(mask)
            mask_list_array.append(mask_array)

        img_list_final, mask_list_final = [], []
        for image ,mask in zip(img_list_array ,mask_list_array):
            img_list_final.append(image)
            mask_list_final.append(mask)
            # image augmentation
            img_aug, mask_aug = augment_seq(image ,mask)
            img_list_final.append(img_aug)
            mask_list_final.append(mask_aug)

        self.imgs = img_list_final
        self.masks = mask_list_final

    def __getitem__(self, idx):
        # load images ad masks
        img = self.imgs[idx]
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = self.masks[idx]

        # mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        num_objs = len(obj_ids)

        # if image contains no object
        if num_objs == 1:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            masks = torch.zeros((1, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            labels = torch.zeros((1,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        else:
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin < xmax and ymin <ymax:
                    boxes.append([xmin, ymin, xmax, ymax])

            # after image augmentation (cropping and scaling), bounding boxes are changed, and there width or height may become 0
            if boxes == []:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                masks = torch.zeros((1, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
                labels = torch.zeros((1,), dtype=torch.int64)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                # there is only one class
                labels = torch.ones((num_objs,), dtype=torch.int64)
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                # suppose all instances are not crowd
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = img.copy()  # cannot accept negative stride
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# For non-RTS images
class PennFudanDataset_train_class0(torch.utils.data.Subset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        img_dir = os.path.join(root, "split_256_nrg/split_images_train_class0")
        img_list = []
        for file in os.listdir(img_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                img_list.append(file)
        img_list = list(sorted(img_list))

        img_list_final = []
        for img_name in img_list:
            img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
            img_list_final.append(np.array(img))
        self.imgs = img_list_final

        mask_dir = os.path.join(root, "split_256_nrg/split_labels_train_class0")
        mask_list = []
        for file in os.listdir(mask_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                mask_list.append(file)
        mask_list = list(sorted(mask_list))

        mask_list_final = []
        for mask_name in mask_list:
            mask = Image.open(os.path.join(mask_dir, mask_name))
            mask_list_final.append(np.array(mask))
        self.masks = mask_list_final

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        boxes = torch.zeros((0, 4), dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        masks = torch.zeros((1, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        labels = torch.zeros((1,), dtype=torch.int64)
        iscrowd = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# For RTS validation images
class PennFudanDataset_val(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        img_dir = os.path.join(root, "split_256_nrg/split_images_val")
        img_list = []
        for file in os.listdir(img_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                img_list.append(file)
        self.imgs = list(sorted(img_list))

        mask_dir = os.path.join(root, "split_256_nrg/split_labels_val")
        mask_list = []
        for file in os.listdir(mask_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                mask_list.append(file)
        self.masks = list(sorted(mask_list))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "split_256_nrg/split_images_val", self.imgs[idx])
        mask_path = os.path.join(self.root, "split_256_nrg/split_labels_val", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        num_objs = len(obj_ids)

        if num_objs == 1:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            masks = torch.zeros((1, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            labels = torch.zeros(1, dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        else:
            obj_ids = obj_ids[1:]
            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if xmin < xmax and ymin < ymax:
                    boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# For RTS test images
class PennFudanDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        img_dir = os.path.join(root, "split_images_test_256")
        img_list = []
        for file in os.listdir(img_dir):
            if os.path.splitext(os.path.basename(file))[1] == '.png':
                img_list.append(file)
        self.imgs = list(sorted(img_list))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "split_images_test_256", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        target = {}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
