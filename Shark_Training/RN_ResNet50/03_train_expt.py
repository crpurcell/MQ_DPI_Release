#!/usr/bin/env python
import argparse
import os
import sys
import warnings
import keras
import keras.preprocessing.image
import tensorflow as tf
from keras_retinanet import layers
from keras_retinanet import losses
from keras_retinanet import models
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import make_shapes_callback
from keras_retinanet.utils.config import read_config_file
from keras_retinanet.utils.config import parse_anchor_parameters
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
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
    backbone = models.backbone(args.backbone)
    check_keras_version()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())
    if args.config:
        args.config = read_config_file(args.config)
    train_generator, validation_generator =                            create_generators(args, backbone.preprocess_image)
    if args.snapshot is not None:
        print("[INFO] loading model, this may take some time  ...")
        model = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        anchor_params = None
        if args.config and "anchor_parameters" in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        prediction_model = retinanet_bbox(model=model,
                                          anchor_params=anchor_params)
    else:
        print("[INFO] creating model, this may take some time ...")
        weights = args.weights
        if weights is None and args.imagenet_weights:
            print("[INFO] fetching ImageNet weights ...")
            weights = backbone.download_imagenet()
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            config=args.config
        )
    print(model.summary())
    callbacks = create_callbacks(model,
                                 training_model,
                                 prediction_model,
                                 validation_generator,
                                 args)
    training_model.fit_generator(generator=train_generator,
                                 steps_per_epoch=args.steps,
                                 epochs=args.epochs,
                                 verbose=1,
                                 callbacks=callbacks)
def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model
def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, config=None):
    modifier = freeze_model if freeze_backbone else None
    anchor_params = None
    num_anchors   = None
    if config and "anchor_parameters" in config:
        anchor_params = parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device("/cpu:0"):
            model = model_with_weights(
                backbone_retinanet(num_classes,
                                   num_anchors=num_anchors,
                                   modifier=modifier),
                weights=weights,
                skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = model_with_weights(
            backbone_retinanet(num_classes,
                               num_anchors=num_anchors,
                               modifier=modifier),
            weights=weights,
            skip_mismatch=True)
        training_model = model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    training_model.compile(
        loss={
            "regression"    : losses.smooth_l1(),
            "classification": losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    return model, training_model, prediction_model
def create_callbacks(model, training_model, prediction_model,
                     validation_generator, args):
    callbacks = []
    tensorboard_callback = None
    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)
    if args.evaluation and validation_generator:
        evaluation = Evaluate(validation_generator,
                              tensorboard=tensorboard_callback,
                              weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)
    if args.snapshots:
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                "{backbone}_{dataset_type}_{{epoch:02d}}.h5"
                .format(backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1,
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = "loss",
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = "auto",
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))
    return callbacks
def create_generators(args, preprocess_image):
    common_args = {
        "batch_size"       : args.batch_size,
        "config"           : args.config,
        "image_min_side"   : args.image_min_side,
        "image_max_side"   : args.image_max_side,
        "preprocess_image" : preprocess_image,
    }
    if args.random_transform:
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
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        **common_args
    )
    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            **common_args
        )
    else:
        validation_generator = None
    return train_generator, validation_generator
def check_args(parsed_args):
    if (parsed_args.multi_gpu > 1 and
        parsed_args.batch_size < parsed_args.multi_gpu):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number " +
            "of GPUs ({})".format(parsed_args.batch_size,
                                  parsed_args.multi_gpu))
    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) " +
            "is not supported.".format(parsed_args.multi_gpu,
                                       parsed_args.snapshot))
    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! " +
            "Run with --multi-gpu-force if you wish to continue.")
    if "resnet" not in parsed_args.backbone:
        warnings.warn(
            "Using experimental backbone {}. Only resnet50 has been " +
            "properly tested.".format(parsed_args.backbone))
    return parsed_args
def parse_args(args):
    ap = argparse.ArgumentParser(
        description="Training a RetinaNet nodel for the Shark AI project.")
    ap.add_argument("-e", "--experiment-path", required=True,
                    help="Path to experiment directory.")
    gp = ap.add_mutually_exclusive_group()
    gp.add_argument("--snapshot",
                    help="Resume training from a snapshot.")
    gp.add_argument("--imagenet-weights",
                    help="Initialize model imagenet weights [default]",
                    action="store_const", const=True, default=True)
    gp.add_argument("--weights",
                    help="Initialize the model with weights from a file.")
    gp.add_argument("--no-weights",
                    help="Don't initialize the model with any weights.",
                    dest="imagenet_weights", action="store_const", const=False)
    ap.add_argument("--backbone",
                    help="Backbone model used by retinanet [resnet50].",
                    default="resnet50", type=str)
    ap.add_argument("--batch-size",
                    help="Size of the batches [1].",
                    default=1, type=int)
    ap.add_argument("--epochs",
                    help="Number of epochs to train [50].",
                    type=int, default=50)
    ap.add_argument("--steps",
                    help="Number of steps per epoch [10,000].",
                    type=int, default=10000)
    ap.add_argument("--image-min-side",
                    help="Rescale image to minimum [1080px] side.",
                    type=int, default=1080) 
    ap.add_argument("--image-max-side",
                    help="Rescale image to maximum [1920px] side.",
                    type=int, default=1920) 
    ap.add_argument("--config",
                    help="Path to a configuration parameters .ini file.")
    ap.add_argument("--gpu",
                    help="ID of the GPU to use (as reported by nvidia-smi).")
    ap.add_argument("--multi-gpu",
                    help="Number of GPUs to use for parallel processing [0].",
                    type=int, default=0)
    ap.add_argument("--multi-gpu-force",
                    help="Extra flag needed to enable multi-gpu support.",
                    action="store_true")
    ap.add_argument("--random-transform",
                    help="Randomly transform image and annotations.",
                    action="store_true")
    ap.add_argument("--no-snapshots",
                    help="Disable saving snapshots.",
                    dest="snapshots", action="store_false")
    ap.add_argument("--no-evaluation",
                    help="Disable per epoch evaluation.",
                    dest="evaluation", action="store_false")
    ap.add_argument("--freeze-backbone",
                    help="Freeze training of backbone layers.",
                    action="store_true")
    ap.add_argument("--weighted-average",
                    help="Compute mAP using  weighted average of classes.",
                    action="store_true")
    return check_args(ap.parse_args(args))
if __name__ == "__main__":
    main()
