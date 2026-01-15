from avl_reference import AVLTreeReference
from avl_array import AVLTreeArray

def main():
    """
    Main function to demonstrate AVL tree visualizations using two implementations:
    1. Reference AVL Tree (pointer-based implementation).
    2. Array-based AVL Tree (implicit representation).

    The function performs the following steps:
    - Initializes a dataset of integers.
    - Constructs and visualizes an AVL tree using the reference implementation.
    - Constructs and visualizes an AVL tree using the array-based implementation.
    """
    data = [10, 20, 30, 40, 50, 25]  # Dataset to be inserted into the AVL trees
    print(f"Visualizing dataset: {data}")

    # Reference AVL Tree Visualization
    print("Generating Reference Tree Visualization...")
    ref = AVLTreeReference()  # Initialize the reference AVL tree
    for x in data:
        ref.root = ref.insert(ref.root, x)  # Insert elements into the reference AVL tree
    ref.visualize("avl_reference_viz")  # Generate and save the visualization

    # Array-based AVL Tree Visualization
    print("Generating Array Tree Visualization...")
    arr = AVLTreeArray()  # Initialize the array-based AVL tree
    for x in data:
        arr.insert(x)  # Insert elements into the array-based AVL tree
    arr.visualize("avl_array_viz")  # Generate and save the visualization

if __name__ == "__main__":
    main()  # Execute the main function