"""
A pre-trained Mask R-CNN model for RTS mapping
Original codes from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Modified by Yiwen lin

Model setting
"""

from torchvision.models.detection import maskrcnn_resnet50_fpn

# Get maskrcnn model pretrained on Imagenet
def maskrcnn_resnet(num_classes, backbone='resnet50', trainable_layers=3):
    # load an instance segmentation model pre-trained on COCO or ImageNet
    # if trainable_backbone_layers is None, pass 3
    model = maskrcnn_resnet50_fpn(pretrained=False, backbone_net=backbone, num_classes=num_classes,
                                  trainable_backbone_layers=trainable_layers)

    return model
