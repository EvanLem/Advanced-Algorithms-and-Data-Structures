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
    sizes = [100, 500, 1000, 2000, 5000]  # Incremental dataset sizes

    ref_times_insert = []
    arr_times_insert = []

    ref_times_search = []
    arr_times_search = []

    ref_mem_peak = []
    arr_mem_peak = []

    for n in sizes:
        print(f"Running experiments for N={n}...")
        dataset = [random.randint(0, 100000) for _ in range(n)]

        # --- Reference AVL ---
        tracemalloc.start()
        start = time.time()
        ref_tree = AVLTreeReference()
        for x in dataset:
            ref_tree.root = ref_tree.insert(ref_tree.root, x)
        ref_times_insert.append(time.time() - start)
        current, peak = tracemalloc.get_traced_memory()
        ref_mem_peak.append(peak / 1024)  # KB
        tracemalloc.stop()

        # Search Ref
        start = time.time()
        for x in dataset:
            ref_tree.search(ref_tree.root, x)
        ref_times_search.append(time.time() - start)

        # --- Array AVL ---
        # Note: Array AVL is much slower due to O(N) data copying.
        # We might limit N for array if it's too slow.
        tracemalloc.start()
        start = time.time()
        arr_tree = AVLTreeArray(capacity=n * 10)  # Pre-allocate
        for x in dataset:
            arr_tree.insert(x)
        arr_times_insert.append(time.time() - start)
        current, peak = tracemalloc.get_traced_memory()
        arr_mem_peak.append(peak / 1024)  # KB
        tracemalloc.stop()

        # Search Array
        start = time.time()
        for x in dataset:
            arr_tree.search(x)
        arr_times_search.append(time.time() - start)

    # --- Plotting ---

    # Time Complexity: Insertion
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ref_times_insert, label='Reference (Pointer) AVL', marker='o')
    plt.plot(sizes, arr_times_insert, label='Array (Implicit) AVL', marker='x')
    plt.title('Insertion Time Complexity')
    plt.xlabel('Number of Elements')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_insert_time.png')

    # Time Complexity: Search
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ref_times_search, label='Reference (Pointer) AVL', marker='o')
    plt.plot(sizes, arr_times_search, label='Array (Implicit) AVL', marker='x')
    plt.title('Search Time Complexity')
    plt.xlabel('Number of Elements')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_search_time.png')

    # Memory Complexity
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ref_mem_peak, label='Reference Peak Memory', marker='o')
    plt.plot(sizes, arr_mem_peak, label='Array Peak Memory', marker='x')
    plt.title('Memory Usage')
    plt.xlabel('Number of Elements')
    plt.ylabel('Memory (KB)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results_memory.png')

    print("Done! Check results_*.png files.")


if __name__ == "__main__":
    run_benchmark()