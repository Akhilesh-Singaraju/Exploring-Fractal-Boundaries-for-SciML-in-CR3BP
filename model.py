import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

num_ensembles = 5

C_VALUES = [3.00, 3.05, 3.10, 3.14, 3.20]

EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

VAL_FRACTION = 0.15
PATIENCE = 7


class BasinClassifier(nn.Module):
    def __init__(self, input_dim=2, num_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),

            nn.Linear(64, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train_ensemble_member(
    c_val,
    model_idx,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    val_fraction=VAL_FRACTION,
    patience=PATIENCE
):


    seed = 42 + int(c_val * 100) + model_idx

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    csv_path = f"files/basin_C{c_val:.2f}.csv"
    df = pd.read_csv(csv_path)

    df_clean = df[~df["label_id"].isin([0, 4])].copy()

    X_raw = df_clean[["x0", "vx0"]].values
    y_raw = df_clean["label_id"].values


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    num_classes = len(label_encoder.classes_)

    print(f"  Classes: {label_encoder.classes_}")


    x_mid = np.median(X_raw[:, 0])
    vx_mid = np.median(X_raw[:, 1])

    test_mask = (
        (X_raw[:, 0] > x_mid) &
        (X_raw[:, 1] > vx_mid)
    )

    X_dev_raw = X_raw[~test_mask]
    y_dev = y_encoded[~test_mask]

    X_test_raw = X_raw[test_mask]
    y_test = y_encoded[test_mask]


    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_dev_raw,
        y_dev,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_dev
    )


    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)


    print(
        f"  Samples: "
        f"train={len(X_train)}, "
        f"val={len(X_val)}, "
        f"test={len(X_test)}"
    )


    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )


    train_dataset = data.TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = data.TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    test_dataset = data.TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = BasinClassifier(
        input_dim=2,
        num_classes=num_classes
    ).to(device)

    class_counts = np.bincount(
        y_train,
        minlength=num_classes
    )

    class_weights = (
        len(y_train)
        / (num_classes * class_counts)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    print("  Training class counts:")
    for class_idx, count in enumerate(class_counts):
        print(
            f"    Class {class_idx}: "
            f"{count} ({count / len(y_train):.2%})"
        )

    print("  Class weights:")
    for class_idx, weight in enumerate(class_weights.cpu().numpy()):
        print(
            f"    Class {class_idx}: "
            f"{weight:.4f}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    test_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):


        model.train()

        running_train_loss = 0.0
        train_correct = 0
        train_samples = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            batch_size_actual = inputs.size(0)

            running_train_loss += (
                loss.item() * batch_size_actual
            )

            train_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            train_samples += batch_size_actual


        train_loss = running_train_loss / train_samples
        train_accuracy = train_correct / train_samples


        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )
        


        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1


        if (epoch + 1) % 5 == 0 or epoch == 0:

            print(
                f"  [C={c_val:.2f} | "
                f"Model {model_idx+1}/{num_ensembles}] "
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        if epochs_without_improvement >= patience:

            print(
                f"  Early stopping at epoch {epoch+1} "
                f"(best val loss = {best_val_loss:.4f})"
            )

            break


    model.load_state_dict(best_state)
    model.to(device)


    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        test_criterion
    )

    print(
        f"  Best Val Loss: {best_val_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )


    return {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "class_weights": class_weights.cpu().numpy(),
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


for C in C_VALUES:

    save_dir = f"files/models/{C:.2f}"
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"\n==============================="
        f"==============================="
    )
    print(
        f"Training Ensemble for C = {C:.2f}"
    )
    print(
        f"==============================="
        f"==============================="
    )


    for i in range(num_ensembles):

        result = train_ensemble_member(
            c_val=C,
            model_idx=i
        )

        model = result["model"]


        model_path = os.path.join(
            save_dir,
            f"MODEL_{i}.pt"
        )

        state_dict_cpu = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

        torch.save(
            state_dict_cpu,
            model_path
        )


        preprocessing_path = os.path.join(
            save_dir,
            f"MODEL_{i}_preprocessing.pt"
        )

        torch.save(
            {
                "scaler_mean": result["scaler"].mean_,
                "scaler_scale": result["scaler"].scale_,
                "label_classes": result["label_encoder"].classes_,
                "class_weights": result["class_weights"],
                "best_val_loss": result["best_val_loss"],
                "test_loss": result["test_loss"],
                "test_accuracy": result["test_accuracy"],
            },
            preprocessing_path
        )


        print(f"Saved model to: {model_path}")
        print(
            f"Saved preprocessing to: "
            f"{preprocessing_path}"
        )