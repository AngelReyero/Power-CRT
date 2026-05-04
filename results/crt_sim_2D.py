# plot_and_save.py

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_results(setting, model, base_dir="csv/sim_2D"):
    pattern = os.path.join(
        base_dir,
        f"simulation2D_{setting}_model{model}_seed*.csv"
    )

    files = glob.glob(pattern)

    if len(files) == 0:
        raise ValueError(f"No files found for pattern: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        if "T_b" in df.columns:
            df = df.drop(columns=["T_b"])

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def plot_and_save(setting, model, save_dir="plots/sim_2D"):
    df = load_results(setting, model)
    # --- Create figure (2x1) ---
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # --- Top: p-values with variance ---
    sns.lineplot(
        data=df,
        x="n",
        y="p_value",
        hue="method",
        marker="o",
        errorbar="sd",   # <-- THIS adds variance bands
        ax=axes[0]
    )
    axes[0].set_title("P-value")
    axes[0].set_ylabel("p-value")
    axes[0].axhline(0.05, linestyle="--", linewidth=1)  # optional reference line

    # --- Bottom: time (log scale) ---
    sns.lineplot(
        data=df,
        x="n",
        y="time",
        hue="method",
        marker="o",
        errorbar="sd",
        ax=axes[1],
        legend=False
    )
    axes[1].set_title("Time (log scale)")
    axes[1].set_ylabel("time")
    axes[1].set_xlabel("n")
    axes[1].set_yscale("log")   # <-- log scale here

    plt.tight_layout()

    # --- Save ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"plot_{setting}_model{model}.pdf"
    )

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {save_path}")

def main():
    setting = "gaussian_linear"
    model = "lasso"

    plot_and_save(setting, model)


def main():
    settings = [
        "gaussian_linear",
        "nonlinear_cos",
        "interaction",
        "heteroskedastic",
    ]

    models = [
        'lm',
        "lasso",
        "RF",
        "GB",
        "NN",
        "SL",
    ]

    for setting in settings:
        for model in models:
            try:
                print(f"Processing: setting={setting}, model={model}")

                plot_and_save(setting, model)

            except Exception as e:
                print(f"Skipping {setting} - {model}: {e}")
                continue

if __name__ == "__main__":
    main()

