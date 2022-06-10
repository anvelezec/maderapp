import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from tqdm import tqdm
import numpy as np

path = "."
model = "RestNet"
val_img = pd.read_csv("images_validation_encoded.csv", header=None)
# pred_img = pd.read_csv("model/validation-anna.csv")
pred_img = pd.read_csv("results/prediction_RestNet.txt", header=None)
val_img.iloc[:, 1] = val_img.iloc[:, 1].str.strip()

y_true, y_pred = [], []
for row, item in tqdm(pred_img.iterrows()):
    file = item[0].split("/")
    file = file if len(file) == 1 else file[1]
    x = val_img[val_img.iloc[:, 1] == file]
    y_true.append(x.iloc[0, 2].strip())
    y_pred.append(item[1].strip())

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")

with open(f"{path}/results_{model}.txt", "w") as file:
    file.write(f"F1, Presicion, Recall, Acc \n")
    file.write(f"{f1}, {prec}, {recall}, {acc} \n")


def display_confusion_matrix(cmat, score, precision, recall, titlestring, img_name):
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    ax.matshow(cmat, cmap="Reds")
    ax.set_xticks(np.array(list(range(cmat.shape[0]))) + 1)
    ax.set_xticklabels(
        np.array(list(range(cmat.shape[0]))) + 1, fontdict={"fontsize": 18}
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(np.array(list(range(cmat.shape[0]))) + 1)
    ax.set_yticklabels(
        np.array(list(range(cmat.shape[0]))) + 1, fontdict={"fontsize": 18}
    )
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if score is not None:
        titlestring += "F1 = {:.3f} ".format(score)
    if precision is not None:
        titlestring += "- Precision = {:.3f} ".format(precision)
    if recall is not None:
        titlestring += "- Recall = {:.3f} ".format(recall)
    if len(titlestring) > 0:
        ax.text(
            20,
            -3,
            titlestring,
            fontdict={
                "fontsize": 30,
                "horizontalalignment": "right",
                "verticalalignment": "top",
                "color": "#804040",
            },
        )
    plt.savefig(img_name)


cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
pd.DataFrame(cm).to_csv(f"{path}/cm_{model}.csv")
img = display_confusion_matrix(
    cm, f1, prec, recall, "", img_name=f"{path}/cm_{model}.png"
)
