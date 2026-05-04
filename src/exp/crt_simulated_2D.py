import numpy as np
from utils import sample_X_tilde_theoretical, generate_simulated_data2D, mse_loss, loss_chooser, prediction_chooser, get_base_model
from crt import T_cv, T_HRT, T_jk, T_OLS, T_trainScore, CRT_comparison
import pandas as pd
import argparse
import os


from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="CRT 2D")
    parser.add_argument("--setting", type=str, default="gaussian_linear", help="Path to real dataset CSV")
    parser.add_argument("--model", type=str,  default="lasso",help="model")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()
  

def main(args):
    s = args.seed
    setting = args.setting
    model_class = get_base_model(args.model, s)
    loss_fn = loss_chooser(setting)
    predict_fn = prediction_chooser(loss_fn)
    n_samples = [30, 50, 70, 100]
    B = 500
    n_jobs = 10

    rng = np.random.default_rng(s)

    # --- Define T functions ---
    T_list = [
        ("ols", lambda X, Y, Z: T_OLS(X, Y, Z)),
        ("hrt", lambda X, Y, Z, model: T_HRT(X, Y, Z, model, predict_fn, loss_fn)),
        ("jackknife", lambda X, Y, Z: T_jk(X, Y, Z, model_class, predict_fn, loss_fn)),
        ("cross_val", lambda X, Y, Z: T_cv(X, Y, Z, model_class, predict_fn, loss_fn, derandomization=10)),
        ("train_score", lambda X, Y, Z: T_trainScore(X, Y, Z, model_class, predict_fn, loss_fn)),
    ]

    all_results = []

    for n in n_samples:
        # --- Generate data ---
        print(f'number of samples:{n}')
        X, Y, Z = generate_simulated_data2D(setting=setting, n=n, seed=s)

        # --- Sample X_tilde ---
        X_tilde = sample_X_tilde_theoretical(Z, B, rng)

        # --- Run CRT ---
        results = CRT_comparison(
            X,
            X_tilde,
            Y,
            Z,
            T_list,
            n_jobs=n_jobs,
            model_HRT=model_class,
            seed=s,
        )

        # --- Store results ---
        for name, res in results.items():
            all_results.append({
                "method": name,
                "n": n,
                "T_orig": res["T_orig"],
                "p_value": res["p_value"],
                "B": B,
                "time": res["time"],
                "T_b": res["T_b"]
            })
    # --- Convert to DataFrame ---
    f_res = pd.DataFrame(all_results)

    # --- Save once at the end ---
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            f"../../results/csv/sim_2D/simulation2D_{setting}_model{args.model}_seed{s}.csv",
        )
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    f_res.to_csv(csv_path, index=False)

    return f_res


# This is the main entry point of the script. It will be executed when the script is 
# run directly, i.e. `python python_script.py --seeds 1 2 3`.
if __name__ == "__main__":
    args = parse_args()
    main(args)




