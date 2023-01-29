import os,sys
from sre_parse import FLAGS
import json
import math
import time
import numpy as np
import random
import argparse
import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from engine import train_one_epoch, evaluate

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

import utils
import transforms as T
from torchvision.transforms import functional as F

from image_augmentation import augment_seq

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data',
                    help='Data file directory, contains both image and mask')

parser.add_argument('--batch_size', type=int, default=2,
                    help='Training batch size')

parser.add_argument('--num_epochs', type=int, default=10,
                    help='Training epochs')

parser.add_argument('--export_dir', type=str, default='exp',
                    help='Export model directory')                   

parser.add_argument('--ckpt_path', type=str, default=None,
                    help='Checkpoint')   

parser.add_argument('--num_workers', type=int, default=0,
                    help='CPU numbers')   

parser.add_argument('--trainable_layers', type=int, default=3,
                    help='Trainable layers of backbone')   

parser.add_argument('--backbone', type=str, default='resnet50',
                    help='Backbone for maskrcnn')


_NUM_CLASSES = 2

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
        for image,mask in zip(img_list_array,mask_list_array):
           img_list_final.append(image)
           mask_list_final.append(mask)
           #image augmentation
           img_aug, mask_aug = augment_seq(image,mask)
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
            if xmin < xmax and ymin<ymax:
              boxes.append([xmin, ymin, xmax, ymax])
          
          #after image augmentation (cropping and scaling), bounding boxes are changed, and there width or height may become 0
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
          img = img.copy() #cannot accept negative stride
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
        img_dir = os.path.join(root, "split_images_test_256_5")
        img_list = []
        for file in os.listdir(img_dir):
           if os.path.splitext(os.path.basename(file))[1] == '.png':
              img_list.append(file)
        self.imgs = list(sorted(img_list))

        mask_dir = os.path.join(root, "split_labels_test_256_5")
        mask_list = []
        for file in os.listdir(mask_dir):
           if os.path.splitext(os.path.basename(file))[1] == '.png':
              mask_list.append(file)
        self.masks = list(sorted(mask_list))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "split_images_test_256_5", self.imgs[idx])
        mask_path = os.path.join(self.root, "split_labels_test_256_5", self.masks[idx])

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
            boxes.append([xmin, ymin, xmax, ymax])

          boxes = torch.as_tensor(boxes, dtype=torch.float32)
          area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
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

# Get maskrcnn model pretrained on Imagenet/COCO 
def maskrcnn_resnet(num_classes,backbone='resnet50',trainable_layers=3):
    
    # load an instance segmentation model pre-trained on COCO or ImageNet
    # if trainable_backbone_layers is None, pass 3
    model = maskrcnn_resnet50_fpn(pretrained=False,backbone_net=backbone,num_classes=num_classes,
                                  trainable_backbone_layers=trainable_layers)

    return model

# Transform data to tensor
def get_transform():
    transforms = []
    # converts the image into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def plot_loss(train_loss_list,val_loss_list,lr_list,time,out_dir):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(111)

  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.plot(train_loss_list, '-o')
  ax1.plot(val_loss_list, '-o')
  ax1.set_ylim([0,1])

  # ax2 for learning rate; share the same x axis with ax1
  ax2 = ax1.twinx()
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Learning rate')
  ax2.plot(lr_list,'k-.')
  
  # ax3 for time; share the same y axis with ax1
  ax3 = ax1.twiny()
  ax3.set_xlim(0, time/60.0)
  ax3.set_xlabel('Relative training time (minutes)')

  ax1.legend(['Train','Validation'])

  plt.tight_layout()
  plt.title('Train vs Validation loss')
  plt.savefig(out_dir)

def plot_metric(val_AP,val_AR,out_dir):
  fig = plt.figure(figsize=(6,4))
  ax1 = fig.add_subplot(111)
  line1 = ax1.plot(val_AP,'b-o',label='Average Precision')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Average Precision')
  ax1.set_ylim([0,1])

  ax2 = ax1.twinx()
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Average Recall')
  line2 = ax2.plot(val_AR,'b-.',label='Average Recall')
  ax2.set_ylim([0,1])

  lns = line1+line2
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc=0)

  plt.tight_layout()
  plt.title('Validation metric - AP')
  plt.savefig(out_dir)

def seed_torch(seed=0):
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def main(unparsed):
    # torch.manual_seed(0)
    seed_torch()

    # get dataset and defined transformations
    data_path = FLAGS.data_dir
    dataset_train_class1 = PennFudanDataset_train_class1(data_path, get_transform())
    dataset_train_class0 = PennFudanDataset_train_class0(data_path, get_transform())
    dataset_train = torch.utils.data.ConcatDataset([dataset_train_class1, dataset_train_class0])
    dataset_val = PennFudanDataset_val(data_path, get_transform())
    
    batch_size = FLAGS.batch_size
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
       dataset_train, batch_size=batch_size, shuffle=True, num_workers=3,
       collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
       dataset_val, batch_size=1, shuffle=False, num_workers=3,
       collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and RTS
    num_classes = _NUM_CLASSES

    # define the model
    backbone = FLAGS.backbone
    trainable_layers = FLAGS.trainable_layers
    model = maskrcnn_resnet(num_classes, backbone=backbone, trainable_layers=trainable_layers)
    
    # if checkpoint exist
    ckpt_path = FLAGS.ckpt_path
    trained_epochs = 0
    if ckpt_path is not None:
       checkpoint = torch.load(ckpt_path)
       model.load_state_dict(checkpoint['model_state_dict'])
       trained_epochs = checkpoint['epoch']+1
       optimizer = checkpoint['optimizer_state_dict']

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                             momentum=0.7, weight_decay=0.001)
    #optimizer = torch.optim.SGD(params, lr=0.001)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    num_epochs = FLAGS.num_epochs
    exp_dir = FLAGS.export_dir

    if os.path.exists(exp_dir) is False:
      os.mkdir(exp_dir)

    epoch_list = []
    lr_list = []
    train_loss_list,val_loss_list = [],[]
    bbox_AP_list, bbox_AR_list, seg_AP_list, seg_AR_list = [],[],[],[]
    epochs_no_improve = 0
    max_seg_ap = 0

    t1 = time.time()

    for epoch in range(num_epochs):
      print('Epoch: [{}]'.format(epoch))
      epoch_list.append(epoch)
      
      print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
      lr_list.append(optimizer.param_groups[0]['lr'])

      # start training, print training loss
      model.train()

      lr_scheduler_warmup = None
      if epoch == 0 and ckpt_path is None:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler_warmup = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

      train_loss_total = 0
      for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
 
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        train_loss_total += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler_warmup is not None:
            lr_scheduler_warmup.step()
      
      train_loss = train_loss_total/len(data_loader)
      print('train_loss: {}'.format(train_loss))
      train_loss_list.append(train_loss)

      # update the learning rate
      # lr_scheduler.step()
      
      # start validation
      # get validation loss
      with torch.no_grad():
        val_loss_total = 0
        for images, targets in data_loader_val:
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          loss_dict = model(images, targets)
          loss_dict_reduced = utils.reduce_dict(loss_dict)
          losses_reduced = sum(loss for loss in loss_dict_reduced.values())
          loss_value = losses_reduced.item()
          val_loss_total += loss_value
        
        val_loss = val_loss_total/len(data_loader_val)
        print('val_loss: {}'.format(val_loss))
        val_loss_list.append(val_loss)
        
        coco_evaluator=evaluate(model, data_loader_val, device=device)
        seg_eval = coco_evaluator.coco_eval['segm'].stats
        seg_ap = seg_eval[1]
        seg_ar = seg_eval[7]
        seg_AP_list.append(seg_ap)
        seg_AR_list.append(seg_ar)

      # save checkpoint at each epoch end
      ckpt_dir = os.path.join(exp_dir,'checkpoints')
      if os.path.exists(ckpt_dir) is False:
        os.mkdir(ckpt_dir)
      
      if (epoch+trained_epochs+1) % 5 == 0:
        ckpt_file = 'ckpt-' + str(epoch+trained_epochs) + '.pth'
        ckpt_path = os.path.join(ckpt_dir,ckpt_file)
      
        torch.save({
            'epoch': epoch+trained_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
      
      #early stopping
      if seg_ap > max_seg_ap:
        epochs_no_improve = 0
        max_seg_ap = seg_ap
      else:
        epochs_no_improve += 1
      if epoch > 15 and epochs_no_improve == 5:
        print('Early stopping!')
        break
    
    t2 = time.time()
    training_time = t2-t1

    metric_dic = {}
    metric_dic['epoch'] = epoch_list
    metric_dic['seg_AP'] = seg_AP_list
    metric_dic['seg_AR'] = seg_AR_list
    metric_file = os.path.join(exp_dir,'metric.txt')
    with open(metric_file,'w') as f_obj:
        json.dump(metric_dic, f_obj, indent=2)
        #f_obj.write(str(dictionary))

    loss_dic = {}
    loss_dic['epoch'] = epoch_list
    loss_dic['lr'] = lr_list
    loss_dic['train_loss'] = train_loss_list
    loss_dic['val_loss'] = val_loss_list
    loss_file = os.path.join(exp_dir,'loss.txt')
    with open(loss_file,'w') as f_obj:
        json.dump(loss_dic, f_obj, indent=2)

    loss_fig_dir = os.path.join(exp_dir,'loss_curve.jpg')
    seg_metrix_fig_dir = os.path.join(exp_dir,'seg_metrix_curve.jpg')
    plot_loss(train_loss_list,val_loss_list,lr_list,training_time,loss_fig_dir)
    plot_metric(seg_AP_list,seg_AR_list,seg_metrix_fig_dir)

    # save model
    model_path = os.path.join(exp_dir,'model.pth')
    torch.save(model.state_dict(), model_path)




if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)

    
    
    
    
