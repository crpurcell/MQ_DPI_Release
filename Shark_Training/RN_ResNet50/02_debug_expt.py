#!/usr/bin/env python
import argparse
import os
import sys
import cv2
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.transform import random_transform_generator
from keras_retinanet.utils.visualization import draw_annotations
from keras_retinanet.utils.visualization import draw_boxes
from keras_retinanet.utils.anchors import anchors_for_shape
from keras_retinanet.utils.anchors import compute_gt_annotations
from keras_retinanet.utils.config import read_config_file
from keras_retinanet.utils.config import parse_anchor_parameters
def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    ePath = args.experiment_path
    if not os.path.exists(ePath):
        sys.exit("[ERR] experiment path does not exist:\n'{}'".format(ePath))
    args.annotations = os.path.join(ePath, "ann_train.csv")
    args.val_annotations = os.path.join(ePath, "ann_valid.csv")
    args.classes = os.path.join(ePath, "classes.csv")
    args.snapshot_path = os.path.join(ePath, "snapshots")
    args.tensorboard_dir = os.path.join(ePath, "logs")
    args.dataset_type = "csv"
    check_keras_version()
    generator = create_generator(args)
    if args.config:
        args.config = read_config_file(args.config)
    anchor_params = None
    if args.config and "anchor_parameters" in args.config:
        anchor_params = parse_anchor_parameters(args.config)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    if args.loop:
        while run(generator, args, anchor_params=anchor_params):
            pass
    else:
        run(generator, args, anchor_params=anchor_params)
def create_generator(args):
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )
    generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        config=args.config
    )
    return generator
def run(generator, args, anchor_params):
    for i in range(generator.size()):
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)
        if args.random_transform:
            image, annotations =                    generator.random_transform_group_entry(image, annotations)
        if args.resize:
            image, image_scale = generator.resize_image(image)
            annotations['bboxes'] *= image_scale
        anchors = anchors_for_shape(image.shape, anchor_params=anchor_params)
        try:
            positive_indices, _, max_indices =                        compute_gt_annotations(anchors, annotations['bboxes'])
        except Exception:
            positive_indices = []
            max_indices = []
        if args.anchors and len(positive_indices) > 0:
            draw_boxes(image, anchors[positive_indices], (255, 255, 0),
                       thickness=1)
        if args.annotations and len(positive_indices) > 0:
            draw_annotations(image, annotations, color=(0, 0, 255),
                             label_to_name=generator.label_to_name)
            draw_boxes(image,
                       annotations['bboxes'][max_indices[positive_indices], :],
                       (0, 255, 0))
        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            return False
    return True
def parse_args(args):
    ap = argparse.ArgumentParser(
        description="Debugging a RetinaNet dataset for the Shark AI project.")
    ap.add_argument("-e", "--experiment-path", required=True,
                    help="Path to experiment directory.")
    ap.add_argument('-l', '--loop', action='store_true',
                    help='Loop forever, even if the dataset is exhausted.')
    ap.add_argument('--no-resize', dest='resize', action='store_false',
                    help='Disable image resizing.')
    ap.add_argument('--anchors', action='store_true',
                    help='Show positive anchors on the image.')
    ap.add_argument('--annotations', action='store_true',
                    help="Show annotations on the image. " +
                    "Green annotations have anchors, red annotations  " +
                    "don't'and therefore don't contribute to training.")
    ap.add_argument('--random-transform', action='store_true',
                    help='Randomly transform image and annotations.')
    ap.add_argument("--image-min-side",
                    help="Rescale image to minimum [1080px] side.",
                    type=int, default=1080) 
    ap.add_argument("--image-max-side",
                    help="Rescale image to maximum [1920px] side.",
                    type=int, default=1920) 
    ap.add_argument("--config",
                    help="Path to a configuration parameters .ini file.")
    return ap.parse_args(args)
if __name__ == "__main__":
    main()
