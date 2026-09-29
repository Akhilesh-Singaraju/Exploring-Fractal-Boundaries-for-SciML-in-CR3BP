import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


C_VALUES = [3.00, 3.05, 3.10, 3.14, 3.20]


def print_distribution(name, y, label_encoder):
    counts = np.bincount(
        y,
        minlength=len(label_encoder.classes_)
    )

    percentages = counts / len(y)

    print(f"\n{name}")
    for cls, count, pct in zip(
        label_encoder.classes_,
        counts,
        percentages
    ):
        print(
            f"  Class {cls}: "
            f"{count:6d} ({pct:.2%})"
        )

    majority_baseline = percentages.max()

    print(
        f"  Majority-class baseline: "
        f"{majority_baseline:.4f}"
    )


for C in C_VALUES:

    print("\n" + "=" * 60)
    print(f"C = {C:.2f}")
    print("=" * 60)

    csv_path = f"files/basin_C{C:.2f}.csv"
    df = pd.read_csv(csv_path)

    df_clean = df[
        ~df["label_id"].isin([0, 4])
    ].copy()

    X_raw = df_clean[["x0", "vx0"]].values
    y_raw = df_clean["label_id"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    x_mid = np.median(X_raw[:, 0])
    vx_mid = np.median(X_raw[:, 1])

    test_mask = (
        (X_raw[:, 0] > x_mid) &
        (X_raw[:, 1] > vx_mid)
    )

    X_dev = X_raw[~test_mask]
    y_dev = y_encoded[~test_mask]

    X_test = X_raw[test_mask]
    y_test = y_encoded[test_mask]

    print_distribution(
        "ALL DATA",
        y_encoded,
        label_encoder
    )

    print_distribution(
        "DEVELOPMENT",
        y_dev,
        label_encoder
    )

    print_distribution(
        "TEST / UPPER-RIGHT",
        y_test,
        label_encoder
    )