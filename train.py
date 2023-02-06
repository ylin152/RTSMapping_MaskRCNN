"""
A pre-trained Mask R-CNN model for RTS mapping
Original codes from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
and https://github.com/pytorch/vision/tree/main/references/detection
Modified by Yiwen lin

Model training
"""

import os,sys
import json
import math
import time
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import DatasetLoader

from model import maskrcnn_resnet
from val import evaluate

import utilities.utils as utils
import utilities.transforms as T


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
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False

def main(unparsed):
    # torch.manual_seed(0)
    seed_torch()

    # get dataset and defined transformations
    data_path = FLAGS.data_dir
    dataset_train_class1 = DatasetLoader.PennFudanDataset_train_class1(data_path, get_transform())
    dataset_train_class0 = DatasetLoader.PennFudanDataset_train_class0(data_path, get_transform())
    dataset_train = torch.utils.data.ConcatDataset([dataset_train_class1, dataset_train_class0])
    dataset_val = DatasetLoader.PennFudanDataset_val(data_path, get_transform())
    
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
        
        coco_evaluator = evaluate(model, data_loader_val, device=device)
        seg_eval = coco_evaluator.coco_eval['segm'].stats
        seg_ap = seg_eval[1]
        seg_AP_list.append(seg_ap)

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
    metric_file = os.path.join(exp_dir,'metric.txt')
    with open(metric_file,'w') as f_obj:
        json.dump(metric_dic, f_obj, indent=2)

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

    
    
    
    
