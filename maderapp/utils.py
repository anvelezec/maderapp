import tensorflow as tf
import torch


def pytorch_model_to_mobile(model_class, num_classes, load_path, save_path):
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, save_path)

def pytorch_model_trace_to_mobile(model_class, num_classes, load_path, save_path):
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    input = torch.rand(1, 3, 224, 224)
    scriptedm = torch.jit.trace(model, input)
    torch.jit.save(scriptedm, save_path)

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


if __name__ == "__main__":
    from maderapp.timber_clasification_efficientNetNS import TimberEfficientNetNS

    """pytorch_model_to_mobile(
        model_class=TimberEfficientNetNS,
        num_classes=25,
        load_path="model_checkpoint/0/efficientNet/maderapp-epoch=249-val_loss=0.02.ckpt",
        save_path="mobile_models/efficientNet.pt",
    )"""

    """pytorch_model_to_mobile(
        model_class=TimberEfficientNetNS,
        num_classes=25,
        load_path="model_checkpoint/0/efficientNet/maderapp-epoch=249-val_loss=0.02.ckpt",
        save_path="mobile_models/trace_efficientNet.pt",
    )"""

    """tensorflow_model_to_mobile(
        load_path="anna/best_weights_fold_5_anna.h5",
        save_path="mobile_models/annaModel.tflite",
    )"""