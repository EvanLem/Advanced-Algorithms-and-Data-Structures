import numpy as np
import graphviz


class AVLTreeArray:
    """
    Implements an AVL tree using an array-based approach (implicit representation).

    Unlike the pointer-based approach, this implementation maps the tree structure
    onto a contiguous block of memory using mathematical index relationships:
    - Root is at index 0.
    - Left child of node i is at 2*i + 1.
    - Right child of node i is at 2*i + 2.

    Attributes:
        capacity (int): The current allocated size of the array.
        tree (numpy.ndarray): The array storing keys. -1 indicates an empty slot (None).
        heights (numpy.ndarray): The array storing node heights.
    """

    def __init__(self, capacity=10):
        """
        Initializes the AVL tree with a given initial capacity.

        Args:
            capacity (int): The initial size of the memory block. Defaults to 10.
        """
        self.capacity = capacity
        # Initialization: -1 indicates an empty slot
        self.tree = np.full(capacity, -1, dtype=np.int32)
        self.heights = np.zeros(capacity, dtype=np.int32)

    # --- Memory Management ---

    def _resize(self, required_index):
        """
        Resizes the internal arrays to accommodate a specific index.

        Unlike standard dynamic arrays (which often double in size), this method
        uses a 'Strict Allocation' strategy to minimize memory usage, although
        sparsity (holes in the index sequence) still consumes space.

        Args:
            required_index (int): The highest index that needs to be written to.
        """
        if required_index >= self.capacity:
            # Strict allocation: allocate exactly what is needed (+1 for safety)
            new_cap = required_index + 1

            new_tree = np.full(new_cap, -1, dtype=np.int32)
            new_heights = np.zeros(new_cap, dtype=np.int32)

            # Copy existing data to the new memory block
            new_tree[:self.capacity] = self.tree
            new_heights[:self.capacity] = self.heights

            self.tree = new_tree
            self.heights = new_heights
            self.capacity = new_cap

    # --- Utility Methods ---

    def get_height(self, idx):
        """
        Retrieves the height of the node at the specified index.

        Args:
            idx (int): The index of the node.

        Returns:
            int: The height of the node, or 0 if the node is empty (-1) or out of bounds.
        """
        if idx >= self.capacity or self.tree[idx] == -1:
            return 0
        return self.heights[idx]

    def update_height(self, idx):
        """
        Updates the height of a node based on the height of its children.

        Args:
            idx (int): The index of the node to update.
        """
        if idx < self.capacity and self.tree[idx] != -1:
            h_left = self.get_height(2 * idx + 1)
            h_right = self.get_height(2 * idx + 2)
            self.heights[idx] = 1 + max(h_left, h_right)

    def get_balance(self, idx):
        """
        Calculates the balance factor of a node (Left Height - Right Height).

        Args:
            idx (int): The index of the node.

        Returns:
            int: The balance factor.
        """
        if idx >= self.capacity or self.tree[idx] == -1:
            return 0
        return self.get_height(2 * idx + 1) - self.get_height(2 * idx + 2)

    # --- Core Logic: Index Mapping & Data Movement ---

    def _extract_subtree_data(self, root_idx, rel_base=0):
        """
        Recursively extracts a subtree's data into a buffer.

        Because we cannot simply change pointers, we must copy data out to move it.
        We store 'relative indices' to preserve the shape of the subtree regardless
        of its absolute position in the array.

        Args:
            root_idx (int): The absolute index of the subtree root.
            rel_base (int): The virtual relative index (starts at 0).

        Returns:
            list: A list of tuples (relative_index, value, height).
        """
        data = []
        if root_idx >= self.capacity or self.tree[root_idx] == -1:
            return data

        # Capture current node data
        data.append((rel_base, self.tree[root_idx], self.heights[root_idx]))

        # Recursively capture children
        # Left child logic: 2*i + 1
        data.extend(self._extract_subtree_data(2 * root_idx + 1, 2 * rel_base + 1))
        # Right child logic: 2*i + 2
        data.extend(self._extract_subtree_data(2 * root_idx + 2, 2 * rel_base + 2))

        return data

    def _clear_subtree(self, idx):
        """
        Physically clears a subtree from memory (sets values to -1).

        Args:
            idx (int): The root index of the subtree to erase.
        """
        if idx >= self.capacity or self.tree[idx] == -1:
            return

        left = 2 * idx + 1
        right = 2 * idx + 2

        self.tree[idx] = -1
        self.heights[idx] = 0

        self._clear_subtree(left)
        self._clear_subtree(right)

    def _map_index(self, base_idx, rel_idx):
        """
        Calculates the new absolute index for a node based on a destination root.

        This method reconstructs the path (Left/Right moves) from the relative index
        and applies it to the new base index. This allows a subtree to be 'grafted'
        anywhere in the array.

        Args:
            base_idx (int): The new root index where the subtree will be placed.
            rel_idx (int): The relative index of the node within that subtree.

        Returns:
            int: The calculated absolute index in the main array.
        """
        if rel_idx == 0:
            return base_idx

        # 1. Trace the path from the relative node back to 0 (Virtual Root)
        path = []
        curr = rel_idx
        while curr > 0:
            if curr % 2 != 0:  # Odd index = Left Child
                path.append('L')
                curr = (curr - 1) // 2
            else:  # Even index = Right Child
                path.append('R')
                curr = (curr - 2) // 2

        # 2. Apply that path starting from the new Base Index
        target = base_idx
        while path:
            move = path.pop()  # Pop reverses order (Top -> Down)
            if move == 'L':
                target = 2 * target + 1
            else:
                target = 2 * target + 2

        return target

    def _write_subtree_data(self, dest_root_idx, data):
        """
        Writes buffered subtree data to a new location in the array.

        Args:
            dest_root_idx (int): The absolute index where the subtree root should go.
            data (list): The list of tuples (rel_idx, value, height) to write.
        """
        for rel_idx, val, h in data:
            # Calculate the mathematical target position
            target_idx = self._map_index(dest_root_idx, rel_idx)

            # Ensure memory exists
            self._resize(target_idx)

            # Write data
            self.tree[target_idx] = val
            self.heights[target_idx] = h

    # --- Rotations (Heavy Copy Operations) ---

    def right_rotate(self, y_idx):
        """
        Performs a Right Rotation at the given index.

        Unlike the pointer implementation, this physically moves memory.
        Topological change:
             y         ->        x
            / \                 / \
           x   T3              T1  y
          / \                     / \
         T1  T2                  T2 T3

        Args:
            y_idx (int): The index of the pivot node Y.
        """
        # Define source indices
        x_idx = 2 * y_idx + 1       # Left child of Y
        t2_idx = 2 * x_idx + 2      # Right child of X

        # Capture Pivot Values
        val_x = self.tree[x_idx]
        val_y = self.tree[y_idx]

        # 1. Extract subtrees that need to move
        # Buffer A: Subtree T1 (Left of X)
        buffer_A = self._extract_subtree_data(2 * x_idx + 1)
        # Buffer B: Subtree T2 (Right of X)
        buffer_T2 = self._extract_subtree_data(t2_idx)
        # Buffer C: Subtree T3 (Right of Y)
        buffer_C = self._extract_subtree_data(2 * y_idx + 2)

        # 2. Clear the affected area to avoid artifacts
        self._clear_subtree(y_idx)

        # 3. Move Pivots
        # X moves up to Y's old position
        self.tree[y_idx] = val_x

        # Y moves down to the right of the new root
        new_y_pos = 2 * y_idx + 2
        self._resize(new_y_pos)
        self.tree[new_y_pos] = val_y

        # 4. Write back subtrees to new calculated positions
        # T1 stays left of X (new root)
        self._write_subtree_data(2 * y_idx + 1, buffer_A)

        # T3 stays right of Y (which moved)
        self._write_subtree_data(2 * new_y_pos + 2, buffer_C)

        # T2 moves from Right of X to Left of Y
        self._write_subtree_data(2 * new_y_pos + 1, buffer_T2)

        # Update heights
        self.update_height(new_y_pos)
        self.update_height(y_idx)

    def left_rotate(self, x_idx):
        """
        Performs a Left Rotation at the given index.

        Topological change:
             x         ->        y
            / \                 / \
           T1  y               x   T3
              / \             / \
             T2 T3           T1 T2

        Args:
            x_idx (int): The index of the pivot node X.
        """
        # Define source indices
        y_idx = 2 * x_idx + 2       # Right child of X
        t2_idx = 2 * y_idx + 1      # Left child of Y

        # Capture Pivot Values
        val_x = self.tree[x_idx]
        val_y = self.tree[y_idx]

        # 1. Extract subtrees
        buffer_A = self._extract_subtree_data(2 * x_idx + 1)  # T1
        buffer_T2 = self._extract_subtree_data(t2_idx)        # T2
        buffer_C = self._extract_subtree_data(2 * y_idx + 2)  # T3

        # 2. Clear area
        self._clear_subtree(x_idx)

        # 3. Move Pivots
        # Y moves up to X's old position
        self.tree[x_idx] = val_y

        # X moves down to the left
        new_x_pos = 2 * x_idx + 1
        self._resize(new_x_pos)
        self.tree[new_x_pos] = val_x

        # 4. Write back subtrees
        self._write_subtree_data(2 * x_idx + 2, buffer_C)      # T3
        self._write_subtree_data(2 * new_x_pos + 1, buffer_A)  # T1
        self._write_subtree_data(2 * new_x_pos + 2, buffer_T2) # T2

        # Update heights
        self.update_height(new_x_pos)
        self.update_height(x_idx)

    # --- Insertion ---

    def insert(self, key):
        """
        Inserts a key into the AVL tree.

        Args:
            key (int): The value to insert.
        """
        self._insert_rec(0, key)

    def _insert_rec(self, idx, key):
        """
        Recursive helper for insertion.

        Args:
            idx (int): The current index being visited.
            key (int): The value to insert.
        """
        # Ensure array is large enough for this path
        self._resize(idx)

        # 1. Standard BST Insertion
        if self.tree[idx] == -1:
            self.tree[idx] = key
            self.heights[idx] = 1
            return

        if key < self.tree[idx]:
            self._insert_rec(2 * idx + 1, key)
        elif key > self.tree[idx]:
            self._insert_rec(2 * idx + 2, key)
        else:
            return  # Duplicate keys are not allowed

        # 2. Update Height
        self.update_height(idx)

        # 3. Check Balance
        balance = self.get_balance(idx)

        # 4. Rebalance if necessary (Rotations)
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2

        # Left-Left (LL) Case
        if balance > 1 and key < self.tree[left_child]:
            self.right_rotate(idx)
            return

        # Right-Right (RR) Case
        if balance < -1 and key > self.tree[right_child]:
            self.left_rotate(idx)
            return

        # Left-Right (LR) Case
        if balance > 1 and key > self.tree[left_child]:
            # Rotate left child first
            self.left_rotate(left_child)
            # Then rotate current
            self.right_rotate(idx)
            return

        # Right-Left (RL) Case
        if balance < -1 and key < self.tree[right_child]:
            # Rotate right child first
            self.right_rotate(right_child)
            # Then rotate current
            self.left_rotate(idx)
            return

    # --- Search & Visualization ---

    def search(self, key):
        """
        Searches for a key in the array-based tree.

        Args:
            key (int): The value to search for.

        Returns:
            bool: True if found, False otherwise.
        """
        curr = 0
        while curr < self.capacity and self.tree[curr] != -1:
            if key == self.tree[curr]:
                return True
            elif key < self.tree[curr]:
                curr = 2 * curr + 1
            else:
                curr = 2 * curr + 2
        return False

    def visualize(self, filename="avl_array_viz"):
        """
        Generates a Graphviz visualization of the tree.

        Nodes are labeled with their value and their array index.

        Args:
            filename (str): The output filename (without extension).
        """
        dot = graphviz.Digraph(comment='AVL Array')
        for i in range(self.capacity):
            if self.tree[i] != -1:
                # Label format: "Value (idx Index)"
                dot.node(str(i), f"{self.tree[i]} (idx {i})")

                left = 2 * i + 1
                right = 2 * i + 2

                if left < self.capacity and self.tree[left] != -1:
                    dot.edge(str(i), str(left))
                if right < self.capacity and self.tree[right] != -1:
                    dot.edge(str(i), str(right))

        dot.render(filename, view=True)