import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from threadpoolctl import threadpool_limits

# Get the maximum available threads on your CPU
MAX_THREADS = torch.get_num_interop_threads()


def run_benchmark_core(taille_1d, taille_2d):
    """Executes the math operations for a single iteration."""
    # --- 1D Arrays (Element-wise) ---
    np_data = np.random.uniform(0.1, 1.0, size=taille_1d).astype(np.float32)
    start = time.perf_counter()
    _ = (
        np.sin(np_data) ** 2
        + np.cos(np_data) ** 2
        + np.exp(np_data) / np_data
    )
    np_1d_time = time.perf_counter() - start

    pt_data = torch.rand(taille_1d, dtype=torch.float32, device="cpu")
    start = time.perf_counter()
    _ = (
        torch.sin(pt_data) ** 2
        + torch.cos(pt_data) ** 2
        + torch.exp(pt_data) / pt_data
    )
    pt_1d_time = time.perf_counter() - start

    # --- 2D Matrices (Matrix Multiplication) ---
    np_A = np.random.randn(taille_2d, taille_2d).astype(np.float32)
    np_B = np.random.randn(taille_2d, taille_2d).astype(np.float32)
    start = time.perf_counter()
    _ = np.dot(np_A, np_B)
    np_2d_time = time.perf_counter() - start

    pt_A = torch.randn(taille_2d, taille_2d, dtype=torch.float32, device="cpu")
    pt_B = torch.randn(taille_2d, taille_2d, dtype=torch.float32, device="cpu")
    start = time.perf_counter()
    _ = torch.matmul(pt_A, pt_B)
    pt_2d_time = time.perf_counter() - start

    return np_1d_time, pt_1d_time, np_2d_time, pt_2d_time


def execute_suite(num_threads, taille_1d, taille_2d, num_runs=3):
    """Executes the benchmark while strict thread limits are enforced for BOTH libraries."""
    print(
        f"▶ Running benchmark with EXACTLY {num_threads} thread(s) for BOTH NumPy and PyTorch..."
    )

    # We use threadpool_limits to restrict NumPy's underlying BLAS/OpenMP engines,
    # and torch.set_num_threads for PyTorch's internal JIT/CPU engine.
    with threadpool_limits(limits=num_threads, user_api="blas"):
        with threadpool_limits(limits=num_threads, user_api="openmp"):
            torch.set_num_threads(num_threads)

            # Warmup run
            _, _, _, _ = run_benchmark_core(taille_1d, taille_2d)

            np_1d_all, pt_1d_all, np_2d_all, pt_2d_all = [], [], [], []
            for _ in range(num_runs):
                n1, p1, n2, p2 = run_benchmark_core(taille_1d, taille_2d)
                np_1d_all.append(n1)
                pt_1d_all.append(p1)
                np_2d_all.append(n2)
                pt_2d_all.append(p2)

    return {
        "np_1d": np.mean(np_1d_all),
        "pt_1d": np.mean(pt_1d_all),
        "np_2d": np.mean(np_2d_all),
        "pt_2d": np.mean(pt_2d_all),
    }


def plot_results(multi_res, single_res, max_threads):
    """Plots a fair 2x2 comparison bar chart."""
    labels = ["Lists 1D (Elements)", "Matrices 2D (Product)"]
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot 1: Multithreaded (FAIR) ---
    ax1.bar(
        x - width / 2,
        [multi_res["np_1d"], multi_res["np_2d"]],
        width,
        label="NumPy",
        color="#4E79A7",
        edgecolor="black",
    )
    ax1.bar(
        x + width / 2,
        [multi_res["pt_1d"], multi_res["pt_2d"]],
        width,
        label="PyTorch",
        color="#F28E2B",
        edgecolor="black",
    )
    ax1.set_title(
        f"Multithreaded mode\n(Each lib uses    {max_threads} threads)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Execution Time (seconds)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # --- Plot 2: Single Threaded (FAIR) ---
    ax2.bar(
        x - width / 2,
        [single_res["np_1d"], single_res["np_2d"]],
        width,
        label="NumPy",
        color="#A0CBE8",
        edgecolor="black",
    )
    ax2.bar(
        x + width / 2,
        [single_res["pt_1d"], single_res["pt_2d"]],
        width,
        label="PyTorch",
        color="#FFBE7D",
        edgecolor="black",
    )
    ax2.set_title(
        "Single-threaded mode\n(Each lib uses 1 thread)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        "CPU benchmark : NumPy vs PyTorch (Thread-to-Thread)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Settings (Sizes optimized for an observable multi-thread scaling effect)
    TAILLE_1D = 20_000_000
    TAILLE_2D = 4000
    RUNS = 3

    print("=" * 60)
    print(f"Starting corrected benchmark (Cores detected: {MAX_THREADS})")
    print("=" * 60)

    # 1. Multi-threaded Run (Both get 24 threads or whatever your MAX_THREADS is)
    res_multi = execute_suite(
        num_threads=MAX_THREADS,
        taille_1d=TAILLE_1D,
        taille_2d=TAILLE_2D,
        num_runs=RUNS,
    )

    # 2. Single-threaded Run (Both get 1 thread)
    res_single = execute_suite(
        num_threads=1, taille_1d=TAILLE_1D, taille_2d=TAILLE_2D, num_runs=RUNS
    )

    print("\nCalculs terminés avec succès ! Génération du graphique...")
    plot_results(res_multi, res_single, MAX_THREADS)