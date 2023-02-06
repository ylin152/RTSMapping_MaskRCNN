"""
A pre-trained Mask R-CNN model for RTS mapping
Original codes from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
and https://github.com/pytorch/vision/tree/main/references/detection
Modified by Yiwen lin

Post-processing & Object based evaluation
"""

import os, sys
import argparse

import basic_src.io_function as io_function
from datasets.remove_mappedPolygons import remove_polygons
from datasets.vector_features import cal_area_length_of_polygon
from datasets.evaluation_result import evaluation_polygons

parser = argparse.ArgumentParser()

parser.add_argument('--remove_area_threshold', type=int, default=900,
                    help='Area threshold for removing post polygons')

parser.add_argument('--inf_dir', type=str, default='inference_output',
                    help='Directory of inference results')

parser.add_argument('--val_file', type=str, default='2017_polygons_south.shp',
                    help='Validation shapefile')

parser.add_argument('--iou_threshold', type=float, default=0.5,
                    help='IOU threshold for evaluation')


def cal_add_area_length_of_polygon(input_shp):
    """
   calculate the area, perimeter of polygons, save to the original file
   :param input_shp: input shapfe file
   :return: True if successful, False Otherwise
   """
    # return vector_features.cal_area_length_of_polygon(input_shp )
    return cal_area_length_of_polygon(input_shp)


def merge(merged_tif):
    if os.path.isfile(merged_tif):
        print('%s already exist' % merged_tif)
    else:
        command_string = 'gdal_merge.py  -init 0 -n 0 -a_nodata 0 -o ' + merged_tif + ' *.tif'
        res = os.system(command_string)
        if res != 0:
            sys.exit(1)


def polygonize(merged_shp, merged_tif):
    if os.path.isfile(merged_shp):
        print('%s already exist' % merged_shp)
    else:
        command_string = 'gdal_polygonize.py -8 %s -b 1 -f "ESRI Shapefile" %s' % (merged_tif, merged_shp)
        res2 = os.system(command_string)
        if res2 != 0:
            sys.exit(1)


def add_attributes(shp_attributes, merged_shp):
    if os.path.isfile(shp_attributes):
        print('%s already exist' % shp_attributes)
    else:
        if io_function.copy_shape_file(merged_shp, shp_attributes) is False:
            raise IOError('copy shape file %s failed' % merged_shp)
        cal_add_area_length_of_polygon(shp_attributes)


def remove_polygons_based_on_area(rm_area_save_shp, shp_attributes):
    if os.path.isfile(rm_area_save_shp):
        print('%s already exist' % rm_area_save_shp)
    else:
        field_name = 'INarea'
        area_threshold = FLAGS.remove_area_threshold
        remove_polygons(shp_attributes, 'INarea', area_threshold, True, rm_area_save_shp)


def evaluate(rm_area_save_shp):
    if os.path.isfile(rm_area_save_shp):
        val_shp = FLAGS.val_file
        val_shp = os.path.join('../data', val_shp)
        iou_threshold = FLAGS.iou_threshold
        out_report = os.path.splitext(os.path.basename(rm_area_save_shp))[0] + '_evaluation_report.txt'
        evaluation_polygons(rm_area_save_shp, val_shp, iou_threshold, out_report)
    else:
        print("shp_post (%s) not exist, stop evaluation" % rm_area_save_shp)


def main(unparsed):
    # test_dir = FLAGS.test_dir
    # inf_dir = os.path.join(test_dir,'inference_output')
    inf_dir = FLAGS.inf_dir
    if not os.path.exists(inf_dir):
        print("inference directory not exist, stop evaluation")
        return
        # os.makedirs(inf_dir)

    os.chdir(inf_dir)
    merged_tif = 'merged.tif'

    # merge patches
    merge(merged_tif)

    # polygonize
    merged_shp = 'merged.shp'
    polygonize(merged_shp, merged_tif)

    # add attributes to shapefile
    shp_attributes = 'merged_post.shp'
    add_attributes(shp_attributes, merged_shp)

    # remove polygons based on area
    rm_area_save_shp = io_function.get_name_by_adding_tail(shp_attributes, 'rmArea')
    remove_polygons_based_on_area(rm_area_save_shp, shp_attributes)

    # evaluate the mapped results
    evaluate(rm_area_save_shp)


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)