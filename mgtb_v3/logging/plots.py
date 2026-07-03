from __future__ import annotations


def plot_logE(logE_values, output_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3))
    plt.plot(list(logE_values))
    plt.xlabel("window")
    plt.ylabel("logE")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
