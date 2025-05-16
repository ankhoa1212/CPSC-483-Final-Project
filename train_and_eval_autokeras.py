import argparse
import os
import time
import autokeras as ak
import keras_tuner
import medmnist
import numpy as np
import tensorflow as tf
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT
from tensorflow.keras.models import load_model


def main(data_flag, num_trials, input_root, output_root, gpu_ids, run, model_path, model_architecture):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    info = INFO[data_flag]
    task = info["task"]
    _ = getattr(medmnist, INFO[data_flag]["python_class"])(
        split="train", root=input_root, download=False
    )

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))

    x_train = npz_file["train_images"]
    y_train = npz_file["train_labels"]
    x_val = npz_file["val_images"]
    y_val = npz_file["val_labels"]
    x_test = npz_file["test_images"]
    y_test = npz_file["test_labels"]

    if model_path is not None:
        model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        test(model, data_flag, x_train, "train", output_root, run)
        test(model, data_flag, x_val, "val", output_root, run)
        test(model, data_flag, x_test, "test", output_root, run)

    if num_trials == 0:
        return

    model = train(
        data_flag, x_train, y_train, x_val, y_val, task, num_trials, output_root, run, model_architecture
    )

    test(model, data_flag, x_train, "train", output_root, run)
    test(model, data_flag, x_val, "val", output_root, run)
    test(model, data_flag, x_test, "test", output_root, run)


def train(
    data_flag, x_train, y_train, x_val, y_val, task, num_trials, output_root, run, model_architecture
):
    clf = None

    if model_architecture:
        # Use custom model architecture
        input_node = ak.ImageInput(np.ndarray([]))
        output_node = ak.Normalization()(input_node)
        output_node1 = ak.ConvBlock()(output_node)
        output_node2 = ak.ResNetBlock(version="v2")(output_node)
        output_node = ak.Merge()([output_node1, output_node2])
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True, 
            metrics=["AUC", "accuracy"], 
            objective=keras_tuner.Objective("val_AUC", direction="max"),
            max_trials=num_trials, 
        )
    else:
        # Use default model architecture
        clf = ak.ImageClassifier(
            multi_label=task == "multi-label, binary-class",
            project_name=data_flag,
            distribution_strategy=tf.distribute.MirroredStrategy(),
            metrics=["AUC", "accuracy"],
            objective=keras_tuner.Objective("val_AUC", direction="max"),
            overwrite=True,
            max_trials=num_trials,
        )

    clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

    model = clf.export_model()

    try:
        model.save(
            os.path.join(output_root, "%s_autokeras_%s" % (data_flag, run)),
            save_format="tf",
        )
    except Exception:
        model.save(
            os.path.join(output_root, "%s_autokeras_%s.keras" % (data_flag, run))
        )

    return model


def test(model, data_flag, x, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)
    y_score = model.predict(x)
    auc, acc = evaluator.evaluate(y_score, output_root, run)
    print("%s  auc: %.5f  acc: %.5f " % (split, auc, acc))

    return auc, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_flag", default="breastmnist", type=str)
    parser.add_argument("--input_root", default=DEFAULT_ROOT, help="root of the directory where *.npz file is read from",type=str)
    parser.add_argument("--output_root", default="./autokeras", type=str)
    parser.add_argument("--model_architecture", default=None, type=str)
    parser.add_argument("--gpu_ids", default="0", type=str)
    parser.add_argument(
        "--run",
        default="model1",
        help="to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="root of the pretrained model to test",
        type=str,
    )
    parser.add_argument(
        "--num_trials",
        default=20,
        help="max_trials of autokeras search space, the script would only test model if num_trials=0",
        type=int,
    )

    args = parser.parse_args()
    data_flag = args.data_flag
    input_root = args.input_root
    output_root = args.output_root
    gpu_ids = args.gpu_ids
    run = args.run
    model_path = args.model_path
    num_trials = args.num_trials
    model_architecture = args.model_architecture

    main(data_flag, num_trials, input_root, output_root, gpu_ids, run, model_path, model_architecture)
