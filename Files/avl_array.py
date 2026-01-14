import numpy as np
import graphviz


class AVLTreeArray:
    def __init__(self, capacity=10):
        self.capacity = capacity
        # Initialisation : -1 indique une case vide
        self.tree = np.full(capacity, -1, dtype=np.int32)
        self.heights = np.zeros(capacity, dtype=np.int32)

    # --- Gestion Mémoire "Stricte" (Non-Gourmande) ---

    def _resize(self, required_index):
        """
        Redimensionne le tableau pour s'adapter EXACTEMENT à l'index demandé.
        Plus d'allocation exponentielle (*2).
        """
        if required_index >= self.capacity:
            # Allocation stricte : on ajoute juste ce qu'il faut (+ un petit buffer de sécurité de 1)
            new_cap = required_index + 1

            new_tree = np.full(new_cap, -1, dtype=np.int32)
            new_heights = np.zeros(new_cap, dtype=np.int32)

            # Copie des données existantes
            new_tree[:self.capacity] = self.tree
            new_heights[:self.capacity] = self.heights

            self.tree = new_tree
            self.heights = new_heights
            self.capacity = new_cap

    # --- Utilitaires AVL ---

    def get_height(self, idx):
        if idx >= self.capacity or self.tree[idx] == -1:
            return 0
        return self.heights[idx]

    def update_height(self, idx):
        if idx < self.capacity and self.tree[idx] != -1:
            h_left = self.get_height(2 * idx + 1)
            h_right = self.get_height(2 * idx + 2)
            self.heights[idx] = 1 + max(h_left, h_right)

    def get_balance(self, idx):
        if idx >= self.capacity or self.tree[idx] == -1:
            return 0
        return self.get_height(2 * idx + 1) - self.get_height(2 * idx + 2)

    # --- Moteur de Déplacement Mathématique ---

    def _extract_subtree_data(self, root_idx, rel_base=0):
        """
        Extrait les données d'un sous-arbre sous forme de liste de tuples.
        Tuple : (index_relatif_virtuel, valeur, hauteur)
        L'index relatif permet de conserver la structure (forme) du sous-arbre.
        """
        data = []
        if root_idx >= self.capacity or self.tree[root_idx] == -1:
            return data

        # On stocke (0, val, h) pour la racine locale, etc.
        data.append((rel_base, self.tree[root_idx], self.heights[root_idx]))

        # Récursion gauche (2*rel + 1) et droite (2*rel + 2)
        data.extend(self._extract_subtree_data(2 * root_idx + 1, 2 * rel_base + 1))
        data.extend(self._extract_subtree_data(2 * root_idx + 2, 2 * rel_base + 2))

        return data

    def _clear_subtree(self, idx):
        """Efface physiquement les données d'un sous-arbre (met à -1)."""
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
        Fonction clé : Calcule le nouvel index absolu.
        Transforme un index relatif (basé sur 0) en index absolu basé sur base_idx.
        Méthode : On décompose le chemin binaire de rel_idx et on l'applique à base_idx.
        """
        if rel_idx == 0:
            return base_idx

        path = []
        curr = rel_idx
        # Remonter le chemin depuis rel_idx jusqu'à 0
        while curr > 0:
            if curr % 2 != 0:  # Impair = enfant gauche
                path.append('L')
                curr = (curr - 1) // 2
            else:  # Pair = enfant droit
                path.append('R')
                curr = (curr - 2) // 2

        # Appliquer le chemin depuis la nouvelle base
        target = base_idx
        while path:
            move = path.pop()  # On dépile pour aller dans le bon sens (Haut vers Bas)
            if move == 'L':
                target = 2 * target + 1
            else:
                target = 2 * target + 2

        return target

    def _write_subtree_data(self, dest_root_idx, data):
        """Réécrit les données extraites à partir d'une nouvelle racine."""
        for rel_idx, val, h in data:
            # Calcul mathématique de la nouvelle position
            target_idx = self._map_index(dest_root_idx, rel_idx)

            # Allocation stricte si nécessaire
            self._resize(target_idx)

            self.tree[target_idx] = val
            self.heights[target_idx] = h

    # --- Rotations ---

    def right_rotate(self, y_idx):
        # Définition des indices sources
        x_idx = 2 * y_idx + 1  # Gauche de Y
        t2_idx = 2 * x_idx + 2  # Droit de X (T2)

        # Valeurs pivots
        val_x = self.tree[x_idx]
        val_y = self.tree[y_idx]

        # 1. Extraction (Sauvegarde des sous-arbres qui vont bouger)
        # On extrait A (Gauche de X), T2 (Droit de X), C (Droit de Y)
        buffer_A = self._extract_subtree_data(2 * x_idx + 1)
        buffer_T2 = self._extract_subtree_data(t2_idx)
        buffer_C = self._extract_subtree_data(2 * y_idx + 2)

        # 2. Nettoyage de la zone
        self._clear_subtree(y_idx)

        # 3. Réécriture (Déplacement des pivots)
        # X monte à la racine (y_idx)
        self.tree[y_idx] = val_x

        # Y descend à droite de la nouvelle racine
        new_y_pos = 2 * y_idx + 2
        self._resize(new_y_pos)
        self.tree[new_y_pos] = val_y

        # 4. Restauration des sous-arbres aux nouvelles positions calculées
        # A reste à gauche de la nouvelle racine
        self._write_subtree_data(2 * y_idx + 1, buffer_A)

        # C reste à droite de Y (qui a bougé)
        self._write_subtree_data(2 * new_y_pos + 2, buffer_C)

        # T2 passe de Droite de X à Gauche de Y
        self._write_subtree_data(2 * new_y_pos + 1, buffer_T2)

        # Mise à jour des hauteurs locales
        self.update_height(new_y_pos)
        self.update_height(y_idx)

    def left_rotate(self, x_idx):
        # Symétrique
        y_idx = 2 * x_idx + 2
        t2_idx = 2 * y_idx + 1

        val_x = self.tree[x_idx]
        val_y = self.tree[y_idx]

        buffer_A = self._extract_subtree_data(2 * x_idx + 1)
        buffer_T2 = self._extract_subtree_data(t2_idx)
        buffer_C = self._extract_subtree_data(2 * y_idx + 2)

        self._clear_subtree(x_idx)

        # Y monte
        self.tree[x_idx] = val_y

        # X descend à gauche
        new_x_pos = 2 * x_idx + 1
        self._resize(new_x_pos)
        self.tree[new_x_pos] = val_x

        self._write_subtree_data(2 * x_idx + 2, buffer_C)
        self._write_subtree_data(2 * new_x_pos + 1, buffer_A)
        self._write_subtree_data(2 * new_x_pos + 2, buffer_T2)

        self.update_height(new_x_pos)
        self.update_height(x_idx)

    # --- Insertion (Logique Standard AVL) ---

    def insert(self, key):
        self._insert_rec(0, key)

    def _insert_rec(self, idx, key):
        # Allocation à la demande
        self._resize(idx)

        # Insertion simple
        if self.tree[idx] == -1:
            self.tree[idx] = key
            self.heights[idx] = 1
            return

        if key < self.tree[idx]:
            self._insert_rec(2 * idx + 1, key)
        elif key > self.tree[idx]:
            self._insert_rec(2 * idx + 2, key)
        else:
            return  # Doublon ignoré

        self.update_height(idx)

        balance = self.get_balance(idx)

        # Cas de rotation (utilisant les indices calculés)
        left = 2 * idx + 1
        right = 2 * idx + 2

        # LL
        if balance > 1 and key < self.tree[left]:
            self.right_rotate(idx)
            return
        # RR
        if balance < -1 and key > self.tree[right]:
            self.left_rotate(idx)
            return
        # LR
        if balance > 1 and key > self.tree[left]:
            self.left_rotate(left)
            self.right_rotate(idx)
            return
        # RL
        if balance < -1 and key < self.tree[right]:
            self.right_rotate(right)
            self.left_rotate(idx)
            return

    def search(self, key):
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
        dot = graphviz.Digraph(comment='AVL Array')
        for i in range(self.capacity):
            if self.tree[i] != -1:
                dot.node(str(i), f"{self.tree[i]} (idx {i})")
                left = 2 * i + 1
                right = 2 * i + 2
                if left < self.capacity and self.tree[left] != -1:
                    dot.edge(str(i), str(left))
                if right < self.capacity and self.tree[right] != -1:
                    dot.edge(str(i), str(right))
        dot.render(filename, view=True)