"""
A pre-trained Mask R-CNN model for RTS mapping
Original codes from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Modified by Yiwen lin

RTS Prediction
"""

import os, sys, json
import argparse
import rasterio
import numpy as np
import torch
import DatasetLoader
from model import maskrcnn_resnet
import utilities.transforms as T

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data',
                    help='Data file directory, contains both image and mask')

parser.add_argument('--img_dir', type=str, default='split_images_test',
                    help='Test images directory, contains test images')

parser.add_argument('--exp_dir', type=str, default='exp',
                    help='Experiment directory')

parser.add_argument('--inf_out_dir', type=str, default='inference_output',
                    help='Output directory')

parser.add_argument('--ckpt_path', type=str,
                    help='Checkpoint for inference')

parser.add_argument('--trainable_layers', type=int, default=2,
                    help='Trainable layers of backbone')

parser.add_argument('--backbone', type=str, default='resnet101',
                    help='Backbone for maskrcnn')

_NUM_CLASSES = 2


# Transform data to tensor
def get_transform():
    transforms = []
    # converts the image into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def main(unparsed):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    output_dir = FLAGS.inf_out_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load model
    backbone = FLAGS.backbone
    trainable_layers = FLAGS.trainable_layers
    model = maskrcnn_resnet(_NUM_CLASSES, backbone=backbone, trainable_layers=trainable_layers)
    exp_dir = FLAGS.exp_dir
    ckpt_path = FLAGS.ckpt_path
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        save_path = os.path.join(exp_dir, 'model.pth')
        model.load_state_dict(torch.load(save_path))

    model.to(device)

    # set model as inference mode
    model.eval()

    # get inference dataset
    data_path = FLAGS.data_dir
    dataset_test = DatasetLoader.PennFudanDataset_test(data_path, get_transform())

    img_dir = FLAGS.img_dir
    img_dir = os.path.join(data_path, img_dir)

    img_files = []

    for file in os.listdir(img_dir):
        if os.path.splitext(os.path.basename(file))[1] == '.png':
            img_files.append(os.path.join(img_dir, file))

    img_files = list(sorted(img_files))

    for (img, _), img_path in zip(dataset_test, img_files):
        pred_img_list = []
        with torch.no_grad():
            prediction = model([img.to(device)])
            # if mask is an empty tensor, there's no RTS in this image, set the predicted image as 0
            if prediction[0]['masks'].nelement() == 0:
                img = torch.zeros_like(img)
                pred_img = img[0].byte().cpu().numpy()
                pred_img_list.append(pred_img)
            else:
                for mask_id in range(prediction[0]['masks'].size(0)):
                    if prediction[0]['scores'][mask_id].cpu().numpy() > 0.5:
                        pred_img = prediction[0]['masks'][mask_id, 0].cpu().numpy()
                        # print('the score of mask_id '+str(mask_id)+' is '+ str(prediction[0]['scores'][mask_id].cpu().numpy()))
                        indices = np.where(np.not_equal(pred_img, 0))
                        for i, j in zip(indices[0], indices[1]):
                            if pred_img[i, j] > 0.5:
                                pred_img[i, j] = 1
                            else:
                                pred_img[i, j] = 0
                        pred_img_list.append(pred_img)

        src_dataset = rasterio.open(img_path)
        image_basename = os.path.splitext(os.path.basename(img_path))[0]
        for img_id, pred_img in enumerate(pred_img_list):
            output_filename = image_basename + '_mask_' + str(img_id + 1) + '.tif'
            path_to_output = os.path.join(output_dir, output_filename)

            with rasterio.open(
                    path_to_output,
                    'w',
                    driver='GTiff',
                    height=src_dataset.shape[0],
                    width=src_dataset.shape[1],
                    count=1,
                    dtype='uint8',
                    transform=src_dataset.transform,
                    crs=src_dataset.crs
            ) as dst_dataset:
                dst_dataset.write(pred_img, 1)


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)