# import tensorflow as tf
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.mobile_optimizer import optimize_for_mobile


def extract_patches(
    image: torch.Tensor, channel: int, kernel_height: int, kernel_width: int
) -> torch.Tensor:
    patches = None
    if image.dim() == 4:
        patches = (
            image.data.unfold(1, channel, channel)
            .unfold(2, kernel_height, kernel_width)
            .unfold(3, kernel_height, kernel_width)
        )
        patches = patches.reshape(-1, channel, kernel_height, kernel_width)
    return patches


def load_model(model_path: str):
    model1 = torch.jit.load(model_path)
    input = torch.rand((1, 3, 224, 224))
    now = datetime.now()
    output = model1(input)
    return output, datetime.now() - now


def show(imgs: list):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def pytorch_model_to_mobile(
    model_class: str, num_classes: int, load_path: str, save_path: str
):
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    model.cpu()
    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, save_path)


def pytorch_model_trace_to_mobile(
    model_class: str, num_classes: int, load_path: str, save_path: str
):
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    model.cpu()
    input = torch.rand(1, 3, 224, 224)
    scriptedm = torch.jit.trace(model, input)
    torch.jit.save(scriptedm, save_path)


def pytorch_model_trace_to_mobile_optim(
    model_class: str, model_params: dict, load_path: str, save_path: str
):

    model = model_class.load_from_checkpoint(load_path, **model_params)
    model.eval()
    model.cpu()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    try:
        traced_script_module = optimize_for_mobile(traced_script_module)
    except RuntimeError as exp:
        print(
            f"Mobile optimization could not be performed, with the following error: {exp}"
        )
    traced_script_module._save_for_lite_interpreter(save_path)


def tensorflow_model_to_mobile(load_path, save_path):

    # Convert the model
    model = tf.keras.models.load_model(load_path)

    # Check its architecture
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(
        model
    )  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open(save_path, "wb") as f:
        f.write(tflite_model)
