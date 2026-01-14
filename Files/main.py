from avl_reference import AVLTreeReference
from avl_array import AVLTreeArray

def main():
    data = [10, 20, 30, 40, 50, 25]
    print(f"Visualizing dataset: {data}")

    # Reference
    print("Generating Reference Tree Visualization...")
    ref = AVLTreeReference()
    for x in data:
        ref.root = ref.insert(ref.root, x)
    ref.visualize("avl_reference_viz")

    # Array
    print("Generating Array Tree Visualization...")
    arr = AVLTreeArray()
    for x in data:
        arr.insert(x)
    arr.visualize("avl_array_viz")

if __name__ == "__main__":
    main()