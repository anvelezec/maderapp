# import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def pytorch_model_to_mobile(model_class, num_classes, load_path, save_path):
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    model.cpu()
    scriptedm = torch.jit.script(model)
    torch.jit.save(scriptedm, save_path)


def pytorch_model_trace_to_mobile(model_class, num_classes, load_path, save_path):
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    model.cpu()
    input = torch.rand(1, 3, 224, 224)
    scriptedm = torch.jit.trace(model, input)
    torch.jit.save(scriptedm, save_path)


def pytorch_model_trace_to_mobile_optim(model_class, num_classes, load_path, save_path):
    import torchvision
    from torch.utils.mobile_optimizer import optimize_for_mobile

    # model = torchvision.models.mobilenet_v2(pretrained=True)
    model = model_class.load_from_checkpoint(load_path, num_classes=num_classes)
    model.eval()
    model.cpu()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(save_path)


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
    from maderapp.timber_clasification_efficientNet import TimberEfficientNet
    from maderapp.timber_clasification_efficientNetNS import TimberEfficientNetNS
    from maderapp.timber_clasification_mobileNet import TimberMobileNet
    from maderapp.timber_clasification_resNet import TimberResNet

    """pytorch_model_to_mobile(
        model_class=TimberEfficientNet,
        num_classes=25,
        load_path="model_checkpoint/None/efficientNet/maderapp-epoch=249-val_loss=0.00.ckpt",
        save_path="mobile_models/efficientNet.pt",
    )

    pytorch_model_trace_to_mobile(
        model_class=TimberEfficientNet,
        num_classes=25,
        load_path="model_checkpoint/None/efficientNet/maderapp-epoch=249-val_loss=0.00.ckpt",
        save_path="mobile_models/trace_efficientNet.pt",
    )
    """
    """    
    models = [TimberEfficientNet, TimberEfficientNetNS, TimberMobileNet, TimberResNet]
    filepaths = ["model_checkpoint/None/efficientNet/maderapp-epoch=249-val_loss=0.00.ckpt",
                 "model_checkpoint/None/efficientNet-NS/maderapp-epoch=249-val_loss=0.01.ckpt",
                 "model_checkpoint/None/MobileNet/maderapp-epoch=225-val_loss=0.01.ckpt",
                 "model_checkpoint/None/RestNet/maderapp-epoch=249-val_loss=0.05.ckpt"]
    savepaths = ["mobile_models/opt_trace_efficientNet.pt", 
                "mobile_models/opt_trace_efficientNetNS.pt", 
                "mobile_models/opt_trace_mobileNet.pt", 
                "mobile_models/opt_trace_restNet.pt"]
    """

    models = [TimberMobileNet]
    filepaths = ["model_checkpoint/datav2/maderapp-epoch=499-val_loss=0.03.ckpt"]
    savepaths = ["mobile_models/opt_trace_mobileNet_v2.pt"]

    for model, filepath, savepath in zip(models, filepaths, savepaths):
        pytorch_model_trace_to_mobile_optim(
            model_class=model,
            num_classes=26,
            load_path=filepath,
            save_path=savepath,
        )

    """tensorflow_model_to_mobile(
        load_path="anna/best_weights_fold_5_anna.h5",
        save_path="mobile_models/annaModel.tflite",
    )"""
