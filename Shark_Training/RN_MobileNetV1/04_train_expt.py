#!/usr/bin/env python
import argparse
import os
import sys
import json
import tensorflow as tf
from google.protobuf import text_format
from object_detection import model_hparams
from object_detection import model_lib
from object_detection.protos import pipeline_pb2
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import save_pipeline_config
from object_detection.utils import label_map_util
def main():
    args = parse_args()
    if args.gpu:
        print("[INFO] setting GPU ID to {}".format(str(args.gpu)))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    recPathJSON = os.path.join(args.experiment_path, "recordpath.json")
    with open(recPathJSON, "r") as fh:
        tmp = json.load(fh)
    recPath = tmp["recordpath"]
    labelMapPath = os.path.join(recPath, "classes.pbtxt")
    labelMap = label_map_util.load_labelmap(labelMapPath)
    numClasses = len(labelMap.item)
    pipeline_config = get_configs_from_pipeline_file(args.config_template)
    pipeline_config = create_pipeline_proto_from_configs(pipeline_config)
    pipeline_config.model.ssd.num_classes = numClasses
    pipeline_config.model.ssd.image_resizer.        fixed_shape_resizer.width = args.image_width
    pipeline_config.model.ssd.image_resizer.        fixed_shape_resizer.height = args.image_height
    pipeline_config.train_config.batch_size = int(args.batch_size)
    pipeline_config.train_config.num_steps = int(args.steps)
    pipeline_config.train_config.optimizer.momentum_optimizer.        learning_rate.cosine_decay_learning_rate.        total_steps = int(args.steps)
    pipeline_config.train_config.optimizer.momentum_optimizer.        learning_rate.cosine_decay_learning_rate.        warmup_steps = int(args.steps * 0.01)
    pipeline_config.graph_rewriter.quantization.delay = int(args.steps / 2)
    pipeline_config.train_config.optimizer.momentum_optimizer.        learning_rate.cosine_decay_learning_rate.        learning_rate_base = args.learning_rate_base
    pipeline_config.train_config.optimizer.momentum_optimizer.        learning_rate.cosine_decay_learning_rate.        warmup_learning_rate = args.learning_rate_base / 50
    weightsDirAbs = os.path.abspath(args.weights_dir)
    weightsPath = os.path.join(weightsDirAbs, "model.ckpt")
    pipeline_config.train_config.fine_tune_checkpoint = weightsPath
    pipeline_config.train_input_reader.label_map_path = labelMapPath
    trainRecPath = os.path.join(recPath, "training.record")
    pipeline_config.train_input_reader.tf_record_input_reader.        input_path[0] = trainRecPath
    pipeline_config.eval_input_reader[0].label_map_path = labelMapPath
    evalRecPath = os.path.join(recPath, "testing.record")
    pipeline_config.eval_input_reader[0].tf_record_input_reader.        input_path[0] = evalRecPath
    save_pipeline_config(pipeline_config, args.experiment_path)
    configPath = os.path.join(args.experiment_path, "pipeline.config")
    modelDir = os.path.join(args.experiment_path, "models")
    runConfig = tf.estimator.RunConfig(model_dir=modelDir)
    trainEvalDict = model_lib.create_estimator_and_inputs(
        run_config=runConfig,
        hparams=model_hparams.create_hparams(hparams_overrides=None),
        pipeline_config_path=configPath,
        sample_1_of_n_eval_examples=1,
    )
    trainSpec, evalSpecs = model_lib.create_train_and_eval_specs(
        train_input_fn = trainEvalDict['train_input_fn'],
        eval_input_fns = trainEvalDict['eval_input_fns'],
        eval_on_train_input_fn = trainEvalDict['eval_on_train_input_fn'],
        predict_input_fn = trainEvalDict['predict_input_fn'],
        train_steps = trainEvalDict['train_steps'],
        eval_on_train_data=False)
    tf.estimator.train_and_evaluate(trainEvalDict['estimator'],
                                    trainSpec, evalSpecs[0])
    tf.app.run()
def parse_args():
    ap = argparse.ArgumentParser(
        description="Training a RetinaNet model for the Shark AI project.")
    ap.add_argument("-e", "--experiment-path", required=True,
                    help="Path to experiment directory.")
    ap.add_argument("-c", "--config-template",
                    default ="templates/default.config",
                    help="Path to TFOD config file [templates/default.config].")
    ap.add_argument("-w", "--weights-dir", default="./weights",
                    help="Path to the starting weights directory [./weights]")
    ap.add_argument("--gpu",
                    help="ID of the GPU to use (as reported by nvidia-smi).")
    ap.add_argument("--learning-rate-base",
                    help="Base learning rate [0.1].",
                    default=0.1, type=float)
    ap.add_argument("--batch-size",
                    help="Size of the batches [8].",
                    default=8, type=int)
    ap.add_argument("--steps",
                    help="Number of steps [30,000].",
                    type=int, default=30000)
    ap.add_argument("--image-width",
                    help="Rescale image to maximum [800px] side.",
                    type=int, default=800)
    ap.add_argument("--image-height",
                    help="Rescale image height [450px] side.",
                    type=int, default=450)
    return ap.parse_args()
if __name__ == "__main__":
    main()
