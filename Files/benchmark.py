"""
This script benchmarks the performance of two AVL tree implementations:
1. Reference AVL Tree (pointer-based implementation).
2. Array-based AVL Tree (implicit representation).

The benchmark measures:
- Insertion time complexity.
- Search time complexity.
- Peak memory usage.

Results are plotted and saved as PNG files for analysis.
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
import sys
from avl_reference import AVLTreeReference
from avl_array import AVLTreeArray
import tracemalloc

# Set recursion limit higher for deep trees
sys.setrecursionlimit(20000)


def run_benchmark():
    """
    Runs the benchmark for both AVL tree implementations and generates plots for:
    - Insertion time complexity.
    - Search time complexity.
    - Memory usage.

    The results are saved as PNG files in the current directory.
    """
    sizes = [100, 500, 1000, 2000, 5000]  # Incremental dataset sizes

    # Lists to store benchmark results
    ref_times_insert = []  # Insertion times for reference AVL
    arr_times_insert = []  # Insertion times for array-based AVL

    ref_times_search = []  # Search times for reference AVL
    arr_times_search = []  # Search times for array-based AVL

    ref_mem_peak = []  # Peak memory usage for reference AVL
    arr_mem_peak = []  # Peak memory usage for array-based AVL

    for n in sizes:
        print(f"Running experiments for N={n}...")
        dataset = [random.randint(0, 100000) for _ in range(n)]  # Generate random dataset

        # --- Reference AVL Tree Benchmark ---
        tracemalloc.start()  # Start memory tracking
        start = time.time()  # Start time tracking
        ref_tree = AVLTreeReference()  # Initialize reference AVL tree
        for x in dataset:
            ref_tree.root = ref_tree.insert(ref_tree.root, x)  # Insert elements
        ref_times_insert.append(time.time() - start)  # Record insertion time
        current, peak = tracemalloc.get_traced_memory()  # Get memory usage
        ref_mem_peak.append(peak / 1024)  # Convert peak memory to KB
        tracemalloc.stop()  # Stop memory tracking

        # Measure search time for reference AVL
        start = time.time()
        for x in dataset:
            ref_tree.search(ref_tree.root, x)  # Search elements
        ref_times_search.append(time.time() - start)

        # --- Array-based AVL Tree Benchmark ---
        # Note: Array-based AVL is slower due to O(N) data copying during insertion.
        tracemalloc.start()  # Start memory tracking
        start = time.time()  # Start time tracking
        arr_tree = AVLTreeArray(capacity=n * 10)  # Initialize array-based AVL with pre-allocated capacity
        for x in dataset:
            arr_tree.insert(x)  # Insert elements
        arr_times_insert.append(time.time() - start)  # Record insertion time
        current, peak = tracemalloc.get_traced_memory()  # Get memory usage
        arr_mem_peak.append(peak / 1024)  # Convert peak memory to KB
        tracemalloc.stop()  # Stop memory tracking

        # Measure search time for array-based AVL
        start = time.time()
        for x in dataset:
            arr_tree.search(x)  # Search elements
        arr_times_search.append(time.time() - start)

    # --- Plotting Results ---

    # Plot insertion time complexity
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ref_times_insert, label='Reference (Pointer) AVL', marker='o')
    plt.plot(sizes, arr_times_insert, label='Array (Implicit) AVL', marker='x')
    plt.title('Insertion Time Complexity')
    plt.xlabel('Number of Elements')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_insert_time.png')  # Save plot as PNG

    # Plot search time complexity
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ref_times_search, label='Reference (Pointer) AVL', marker='o')
    plt.plot(sizes, arr_times_search, label='Array (Implicit) AVL', marker='x')
    plt.title('Search Time Complexity')
    plt.xlabel('Number of Elements')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_search_time.png')  # Save plot as PNG

    # Plot memory usage
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ref_mem_peak, label='Reference Peak Memory', marker='o')
    plt.plot(sizes, arr_mem_peak, label='Array Peak Memory', marker='x')
    plt.title('Memory Usage')
    plt.xlabel('Number of Elements')
    plt.ylabel('Memory (KB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_memory.png')  # Save plot as PNG

    print("Done! Check results_*.png files.")  # Notify user of completion


if __name__ == "__main__":
    run_benchmark()  # Execute the benchmark