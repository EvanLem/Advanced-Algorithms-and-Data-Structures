import graphviz


class Node:
    """
    Represents a node in the AVL tree.

    Attributes:
        key (int): The value stored in the node.
        left (Node): Pointer to the left child node.
        right (Node): Pointer to the right child node.
        height (int): Height of the node in the tree.
    """
    def __init__(self, key):
        """
        Initializes a new node with the given key.

        Args:
            key (int): The value to be stored in the node.
        """
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # Initial height of a leaf node is 1


class AVLTreeReference:
    """
    Implements an AVL tree using a pointer-based approach.

    The AVL tree is a self-balancing binary search tree where the difference
    between the heights of the left and right subtrees cannot be more than one.
    """
    def __init__(self):
        """
        Initializes an empty AVL tree.
        """
        self.root = None

    # --- Utility Methods ---

    def get_height(self, node):
        """
        Returns the height of the given node.

        Args:
            node (Node): The node whose height is to be retrieved.

        Returns:
            int: The height of the node, or 0 if the node is None.
        """
        if not node:
            return 0
        return node.height

    def get_balance(self, node):
        """
        Calculates the balance factor of the given node.

        Args:
            node (Node): The node whose balance factor is to be calculated.

        Returns:
            int: The balance factor (difference between left and right subtree heights).
        """
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def _update_height(self, node):
        """
        Updates the height of the given node based on its children.

        Args:
            node (Node): The node whose height is to be updated.
        """
        if node:
            node.height = 1 + max(self.get_height(node.left),
                                  self.get_height(node.right))

    # --- Rotations (O(1) Complexity) ---
    # Rotations only change pointers; no data is copied or moved in memory.

    def right_rotate(self, y):
        """
        Performs a right rotation on the given subtree.

        Args:
            y (Node): The root of the subtree to be rotated.

        Returns:
            Node: The new root of the rotated subtree.
        """
        x = y.left
        T2 = x.right

        # Perform rotation (pointer exchange)
        x.right = y
        y.left = T2

        # Update heights (update y first as it is moved down)
        self._update_height(y)
        self._update_height(x)

        # x becomes the new root of the subtree
        return x

    def left_rotate(self, x):
        """
        Performs a left rotation on the given subtree.

        Args:
            x (Node): The root of the subtree to be rotated.

        Returns:
            Node: The new root of the rotated subtree.
        """
        y = x.right
        T2 = y.left

        # Perform rotation (pointer exchange)
        y.left = x
        x.right = T2

        # Update heights
        self._update_height(x)
        self._update_height(y)

        # y becomes the new root of the subtree
        return y

    # --- Insertion ---

    def insert(self, root, key):
        """
        Inserts a new key into the AVL tree and rebalances it if necessary.

        Args:
            root (Node): The root of the subtree where the key is to be inserted.
            key (int): The key to be inserted.

        Returns:
            Node: The new root of the subtree after insertion.
        """
        # 1. Perform standard BST insertion
        if not root:
            return Node(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:
            return root  # No duplicates allowed

        # 2. Update the height of the current node
        self._update_height(root)

        # 3. Check the balance factor
        balance = self.get_balance(root)

        # 4. Perform rotations if the node becomes unbalanced

        # Left-Left (LL) Case
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)

        # Right-Right (RR) Case
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)

        # Left-Right (LR) Case
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # Right-Left (RL) Case
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    # --- Deletion (Optional but recommended for completeness) ---

    def get_min_value_node(self, root):
        """
        Finds the node with the smallest key in the subtree.

        Args:
            root (Node): The root of the subtree.

        Returns:
            Node: The node with the smallest key.
        """
        if root is None or root.left is None:
            return root
        return self.get_min_value_node(root.left)

    def delete(self, root, key):
        """
        Deletes a key from the AVL tree and rebalances it if necessary.

        Args:
            root (Node): The root of the subtree where the key is to be deleted.
            key (int): The key to be deleted.

        Returns:
            Node: The new root of the subtree after deletion.
        """
        if not root:
            return root

        # Perform standard BST deletion
        if key < root.key:
            root.left = self.delete(root.left, key)
        elif key > root.key:
            root.right = self.delete(root.right, key)
        else:
            # Node to be deleted found
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left

            # Node with two children: Get the in-order successor
            temp = self.get_min_value_node(root.right)
            root.key = temp.key
            root.right = self.delete(root.right, temp.key)

        if not root:
            return root

        # Update height and rebalance the tree
        self._update_height(root)
        balance = self.get_balance(root)

        # Left-Left (LL) Case
        if balance > 1 and self.get_balance(root.left) >= 0:
            return self.right_rotate(root)
        # Left-Right (LR) Case
        if balance > 1 and self.get_balance(root.left) < 0:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        # Right-Right (RR) Case
        if balance < -1 and self.get_balance(root.right) <= 0:
            return self.left_rotate(root)
        # Right-Left (RL) Case
        if balance < -1 and self.get_balance(root.right) > 0:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    # --- Search and Visualization ---

    def search(self, root, key):
        """
        Searches for a key in the AVL tree.

        Args:
            root (Node): The root of the subtree to search.
            key (int): The key to search for.

        Returns:
            Node: The node containing the key, or None if not found.
        """
        if root is None or root.key == key:
            return root
        if key < root.key:
            return self.search(root.left, key)
        return self.search(root.right, key)

    def visualize(self, filename="avl_reference_viz"):
        """
        Generates a visualization of the AVL tree using Graphviz.

        Args:
            filename (str): The name of the output file (without extension).
        """
        dot = graphviz.Digraph(comment='AVL Reference')

        def add_nodes(node):
            """
            Recursively adds nodes and edges to the Graphviz object.

            Args:
                node (Node): The current node being processed.
            """
            if node:
                dot.node(str(node.key), str(node.key))
                if node.left:
                    dot.edge(str(node.key), str(node.left.key))
                    add_nodes(node.left)
                if node.right:
                    dot.edge(str(node.key), str(node.right.key))
                    add_nodes(node.right)

        if self.root:
            add_nodes(self.root)

        dot.render(filename, view=True)