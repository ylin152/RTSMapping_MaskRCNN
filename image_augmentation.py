import os,sys
import argparse
from unicodedata import name
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

parser = argparse.ArgumentParser()

parser.add_argument('--image_array', type=float, 
                    help='Image array')

parser.add_argument('--mask_array', type=float, 
                    help='Maks array')

parser.add_argument('--augment_list', type=list, 
                    help='Augmentation techniques')


def Flip(image_np, image_list):
    """
    Flip image horizontally and vertically
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        file_basename: File base name (e.g basename.tif)
        is_groud_true:
    Returns: True if successful, False otherwise
    """
    flipper = iaa.Fliplr(1.0)  # always horizontally flip each input image; Fliplr(P) Horizontally flips images with probability P.
    image_lr = flipper.augment_image(image_np)  # horizontally flip image 0
    # image_lr[image_lr < 0] = 0
    #
    vflipper = iaa.Flipud(1.0)  # vertically flip each input image with 90% probability
    image_ud = vflipper.augment_image(image_np)  # probably vertically flip image 1
    image_ud[image_ud < 0] = 0
    
    image_list.append(image_lr)
    # image_list.append(image_ud)

    return image_list

def rotate(image_np, image_list, degree=[90,180,270]):
    """
    roate image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        degree: the degree list for rotation
        is_groud_true:
    Returns: True if successful, False otherwise
    """
    for angle in degree:
        roate = iaa.Affine(rotate=angle)
        image_r = roate.augment_image(image_np)
        # image_r[image_r < 0] = 0
        image_list.append(image_r)

    return image_list

def scale(image_np, image_list, scale=[0.75,1.25]):
    """
    scale image with 90, 180, 270 degree
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        scale: the scale list for zoom in or zoom out
        is_groud_true:
    Returns: True is successful, False otherwise
    """
    for value in scale:
        scale = iaa.Affine(scale=value)
        image_s = scale.augment_image(image_np)
        # image_s[image_s < 0] = 0
        image_list.append(image_s)

    return image_list

def blurer(image_np, image_list, is_ground_true=False, sigma=[1,2]):
    """
    Blur the original images
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        sigma: sigma value for blurring
    Returns: True if successful, False otherwise
    """
    for value in sigma:
        if is_ground_true is True:
           # just copy the groud true
           image_b = image_np
        else:
           blurer = iaa.GaussianBlur(value)
           image_b = blurer.augment_image(image_np)
        #    image_b[image_b < 0] = 0
        image_list.append(image_b)

    return image_list

def brightness(image_np, image_list, is_ground_true=False, out_count=1):
    """
    Change the brightness of images: MultiplyAndAddToBrightness
    :param image_np: 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    :param save_dir: the directory for saving images
    :param input_filename: File base name (e.g basename.tif)
    :param is_groud_true: if ground truth, just copy the image
    :return:
    """
    for idx in range(out_count):
        if is_ground_true is True:
           # just copy the groud true
           image_b = image_np
        else:
           brightness = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))  # a random value between the range
           image_b = brightness.augment_image(image_np)
        #    image_b[image_b < 0] = 0
        image_list.append(image_b)

    return image_list

def contrast(image_np, image_list, is_ground_true=False, out_count=1):
    """
    Change the constrast of images: GammaContrast
    :param image_np: 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    :param save_dir:
    :param input_filename: File base name (e.g basename.tif)
    :param is_groud_true: if ground truth, just copy the image
    :param :
    :return:
    """
    for idx in range(out_count):
        if is_ground_true is True:
           # just copy the groud true
           image_con = image_np
        else:
           contrast = iaa.GammaContrast((0.5, 1.5))  # a random gamma value between the range, a large gamma make image darker
           image_con = contrast.augment_image(image_np)
        #    image_con[image_con < 0] = 0
        image_list.append(image_con)

    return image_list

def noise(image_np, image_list, is_ground_true=False, out_count=1):
    """
    Change the constrast of images: AdditiveGaussianNoise
    :param image_np: 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    :param save_dir:
    :param input_filename: File base name (e.g basename.tif)
    :param is_groud_true: if ground truth, just copy the image
    :param :
    :return:
    """
    for idx in range(out_count):
        if is_ground_true is True:
           # just copy the groud true
           image_noise = image_np
        else:
           noise = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))  # a random gamma value between the range
           image_noise = noise.augment_image(image_np)
        #    image_noise[image_noise < 0] = 0
        image_list.append(image_noise)

    return image_list

def Crop(image_np, image_list, px = [10,30]):
    """
    Crop the original images
    Args:
        image_np: image_np:  'images' should be either a 4D numpy array of shape (N, height, width, channels)
        save_dir: the directory for saving images
        input_filename: File base name (e.g basename.tif)
        px:
        is_groud_true
    Returns: True if successful, False otherwise
    """
    for value in px:
        crop = iaa.Crop(px=value,keep_size=False)
        image_c = crop.augment_image(image_np)
        # image_c[image_c < 0] = 0
        image_list.append(image_c)

    return image_list

def augment(image_array, image_list, augment_list, is_ground_true=False):

    if 'flip' in  augment_list:
        image_list = Flip(image_array,image_list)
    if 'rotate' in augment_list:
        image_list = rotate(image_array,image_list)
    if 'blur' in augment_list:
        image_list = blurer(image_array, image_list, is_ground_true=is_ground_true, sigma=[1, 2])
    if 'crop' in augment_list:
        image_list = Crop(image_array, image_list, px=[10])
    if 'scale' in augment_list:
        image_list = scale(image_array, image_list)
    if 'bright' in augment_list:
        image_list = brightness(image_array, image_list, is_ground_true=is_ground_true, out_count=2)
    if 'contrast' in augment_list:
        image_list = contrast(image_array, image_list, is_ground_true=is_ground_true, out_count=2)
    if 'noise' in augment_list:
        image_list = noise(image_array, image_list, is_ground_true=is_ground_true, out_count=2)

    return image_list

def augment_seq(img_array, mask_array):
    
    ia.seed(1)

    segmap = SegmentationMapsOnImage(mask_array, shape=img_array.shape)

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(1,name='Fliplr'), # horizontally flip 100% of all images
        iaa.Flipud(0.5,name='Flipud'), # vertically flip 50% of all images
        # crop images by -5% to 10% of their height/width
        iaa.Crop(
            percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1]), #negative value means padding, positive value means cropping
            name = 'Crop'
        ),
        sometimes(iaa.Affine(
            scale={"x": (1.0, 1.2), "y": (1.0, 1.2)}, # scale images to 100-120% of their size, individually per axis
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            name = 'Affine'
        )),
        # execute 1 to 2 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((1, 2),
            [
                iaa.GaussianBlur((0, 2.0),name='Blur'),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255),name='Noise'), # add gaussian noise to images
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30),name='Brightness'), # change brightness
                iaa.GammaContrast((0.5, 1.5),name='Contrast'), # improve or worsen the contrast
            ],
            random_order=True
        )
    ],random_order=True
    )

    # change the activated augmenters for mask
    # def activator_mask(image, augmenter, parents, default):
    #     if augmenter.name in ['Blur', 'Noise', 'Brightness','Contrast']:
    #         return False
    #     else:
    #         # default value for all other augmenters
    #         return default
    # hooks_mask = ia.HooksImages(activator=activator_mask)

    img_aug, mask_aug = seq(image=img_array, segmentation_maps=segmap)

    mask_aug = SegmentationMapsOnImage.get_arr(mask_aug)

    return img_aug, mask_aug

def main(unparsed):
    image_array = FLAGS.image_array
    mask_array = FLAGS.mask_array
    augment_list = FLAGS.augment_list

    # augment(image_array, image_list, augment_list)
    augment_seq(image_array,mask_array)

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)