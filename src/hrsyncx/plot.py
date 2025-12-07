# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "jax",
# ]
# ///
from typing import Optional
import json
import matplotlib.pyplot as plt


def plot_fig1(file_path: str = "fig1.json", save_path: Optional[str] = None):
    with open(file_path, "r") as f:
        results = json.load(f)

    p_rs = results["p_rs"][:]
    alphas = results["alphas"][:]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for alpha in alphas:
        sync_errors = results["sync_error_results"][str(alpha)][:]
        ax.plot(
            p_rs,
            sync_errors,
            marker="o",
            linestyle="-",
            label=r"$\alpha = $" + f"{alpha}",
        )
    # ax.set_xscale("log")
    ax.set_xlabel(r"$p_r$", fontsize=15)
    ax.set_ylabel(r"$E$", fontsize=15)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=15,
    )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_fig2(file_path: str, save_path: Optional[str] = None):
    with open(file_path, "r") as f:
        results = json.load(f)

    p_rs = results["p_rs"]
    epsilons = results["epsilon"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for epsilon in epsilons:
        sync_errors = results["sync_error_results"][str(epsilon)]
        ax.plot(
            p_rs,
            sync_errors,
            marker="o",
            linestyle="-",
            label=r"$\epsilon = $" + f"{epsilon}",
        )
    # ax.set_xscale("log")
    ax.set_xlabel(r"$p_r$", fontsize=15)
    ax.set_ylabel(r"$E$", fontsize=15)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=15,
    )

    fig.tight_layout()
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    file_path = "results/fig1-50N.json"
    file_name = file_path.split("/")[1].split(".")[0]
    save_path = f"figures/{file_name}.svg"
    plot_fig1(file_path, save_path)
