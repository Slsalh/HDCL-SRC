import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, deque


class FTMLoss(nn.Module):
    def __init__(self, gamma=2.0, hiera=None, label_dict=None):
        super().__init__()
        self.gamma = gamma
        self.hiera = hiera
        self.label_dict = label_dict
        self.ancestor_cache = {}
        self.descendant_cache = {}
        num_classes = len(self.label_dict)

        # pre-calculate ancestor and descendant categories and cache them
        for v in range(num_classes):
            self.ancestor_cache[v] = self.get_all_ancestors(v)
            self.descendant_cache[v] = self.get_all_descendants(v)

    # get all ancestors
    def get_all_ancestors(self, v):
        ancestors = []
        queue = [v]
        while queue:
            current = queue.pop(0)
            parents = [p for p, children in self.hiera.items() if current in children]
            ancestors.extend(parents)
            queue.extend(parents)
        return list(set(ancestors))  # deduplication

    # get all descendants
    def get_all_descendants(self, v):
        descendants = []
        queue = self.hiera.get(v, [])
        while queue:
            current = queue.pop(0)
            descendants.append(current)
            queue.extend(self.hiera.get(current, []))
        return descendants

    def adjust_probs_with_hierarchy(self, logits: torch.Tensor, labels: torch.Tensor):
        # initial probability
        probs = torch.sigmoid(logits)  # shape: [batch_size, num_classes]
        batch_size, num_classes = probs.shape
        adjusted_probs = probs.clone()

        # iterate through each category
        for v in range(num_classes):
            # dealing with ancestral relationships
            if v in self.ancestor_cache and self.ancestor_cache[v]:
                ancestor_idx = torch.tensor(self.ancestor_cache[v], device=probs.device)
                ancestor_probs = probs[:, ancestor_idx]  # probability of extracting all ancestors
                min_ancestor_probs = ancestor_probs.min(dim=1)[0]  # take the minimum value along the ancestral dimension.
                # when the label is 1, the adjustment probability is the smaller of the self-probability and the minimum ancestor probability.
                adjusted_probs[:, v] = torch.where(labels[:, v] == 1,
                                                   torch.min(probs[:, v], min_ancestor_probs),
                                                   adjusted_probs[:, v])

            # dealing with generational relationships
            if v in self.descendant_cache and self.descendant_cache[v]:
                descendant_idx = torch.tensor(self.descendant_cache[v], device=probs.device)
                descendant_probs = probs[:, descendant_idx]  # probability of extracting all descendants
                max_descendant_probs = descendant_probs.max(dim=1)[0]  # take the maximum value along the descendant dimension.
                # when the label is 0, the adjustment probability is the greater of the self-probability and the maximum probability of the offspring.
                adjusted_probs[:, v] = torch.where(labels[:, v] == 0,
                                                   torch.max(probs[:, v], max_descendant_probs),
                                                   adjusted_probs[:, v])
        return adjusted_probs

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        adjusted_probs = self.adjust_probs_with_hierarchy(logits, labels)
        pos_loss = -labels * (1 - adjusted_probs) ** self.gamma * torch.log(adjusted_probs + 1e-8)
        neg_loss = -(1 - labels) * adjusted_probs ** self.gamma * torch.log(1 - adjusted_probs + 1e-8)
        loss = (pos_loss + neg_loss).sum(dim=1).mean()
        return loss


class HierarchicalTripletLoss(nn.Module):
    def __init__(self, class_hierarchy, label_dict, margin_e=0.2, alpha=0.5, top_k_negatives=5):
        """
        Args:
            class_hierarchy (dict): label hierarchy structure {child: parent}
            label_dict (dict):  {idx: class_name}
            margin_e (float): fixed margins
        """
        super().__init__()
        self.margin_e = margin_e
        self.alpha = alpha
        self.top_k_negatives = top_k_negatives
        self.D = self.calculate_tree_height(class_hierarchy)

        self._build_hierarchy(class_hierarchy, label_dict)

        # compute distance matrix
        self.distance_matrix = self._precompute_distances(label_dict)
        # self.M_g = self._compute_global_margin(label_dict)
        self.num_labels = len(label_dict)

        # compute sibling relationship
        self._precompute_siblings(class_hierarchy)

    def _precompute_siblings(self, class_hierarchy):
        self.sibling_map = defaultdict(set)
        parent_to_children = defaultdict(list)

        # construct a parent to child node mapping
        for child, parent in class_hierarchy.items():
            if child not in self.name2idx:
                continue
            child_idx = self.name2idx[child]
            parent_idx = self.parent_map.get(child_idx, self.root_idx)
            if parent_idx != self.root_idx:  # excluding cases where the parent node is a Root
                parent_to_children[parent_idx].append(child_idx)

        # sibling relationship
        for parent, children in parent_to_children.items():
            for child in children:
                siblings = set(children) - {child}
                self.sibling_map[child].update(siblings)

    def calculate_tree_height(self, class_hierarchy):
        """calculate tree height"""
        children_dict = defaultdict(list)
        for child, parent in class_hierarchy.items():
            children_dict[parent].append(child)
        queue = deque([('Root', 0)])
        max_depth = 0
        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)
            if node in children_dict:
                for child in children_dict[node]:
                    queue.append((child, depth + 1))
        return max_depth

    def _build_hierarchy(self, class_hierarchy, label_dict):
        """build hierarchy relationship"""
        self.name2idx = {v: k for k, v in label_dict.items()}
        self.root_idx = max(label_dict.keys()) + 1
        self.parent_map = {}
        self.children_map = defaultdict(list)

        for child, parent in class_hierarchy.items():
            if child not in self.name2idx:
                continue
            child_idx = self.name2idx[child]
            parent_idx = self.name2idx[parent] if parent in self.name2idx else self.root_idx
            self.parent_map[child_idx] = parent_idx
            self.children_map[parent_idx].append(child_idx)

    def _precompute_distances(self, label_dict):
        """computing semantic distance based on LCA height: d_T = h(lca(u,v)) / h(T)"""
        indices = list(label_dict.keys()) + [self.root_idx]
        n = len(indices)
        dist_matrix = np.zeros((n, n), dtype=np.float32)

        for i, u in enumerate(indices):
            for j, v in enumerate(indices):
                if i == j:
                    continue
                lca = self._find_lca(u, v)
                height_lca = self._get_height(lca)
                dist_matrix[i][j] = height_lca / self.D

        return torch.from_numpy(dist_matrix).float()

    def _get_height(self, node_idx):
        """calculate the depth of the node to the Root"""
        depth = 0
        current = node_idx
        while current in self.parent_map:
            depth += 1
            current = self.parent_map[current]
        return depth

    def _is_ancestor_or_descendant(self, anchor_idx, n_idx):
        """check if n_idx is an ancestor or descendant of anchor_idx"""
        # ancestor
        current = anchor_idx
        while current in self.parent_map:
            if current == n_idx:
                return True
            current = self.parent_map[current]
        # descendant
        if n_idx in self._get_descendants(anchor_idx):
            return True
        return False

    def _get_descendants(self, node_idx):
        """get all descendants of node_idx"""
        descendants = set()
        stack = [node_idx]
        while stack:
            current = stack.pop()
            if current in self.children_map:
                for child in self.children_map[current]:
                    descendants.add(child)
                    stack.append(child)
        return descendants

    def _get_triplets(self, label_ids):
        """generate triples, combining current sample and Batch information"""
        batch_triplets = []
        distance_matrix_np = self.distance_matrix.cpu().numpy()

        # constructing a label-to-sample inverted index
        label_to_batch = defaultdict(list)
        for bid in range(label_ids.size(0)):
            for label_id in torch.where(label_ids[bid])[0].tolist():
                label_to_batch[label_id].append(bid)

        # iterate over each label of each sample to generate triples
        for bid in range(label_ids.size(0)):
            current_labels = torch.where(label_ids[bid])[0].tolist()
            positive_idx = None
            pos_bid = None
            neg_bid = None
            negative_idx = None
            for anchor_idx in current_labels:
                # 1. sibling triplets
                # positive sample selection
                # prioritize sibling nodes from the current sample
                same_sample_siblings = [
                    (bid, other_label)
                    for other_label in current_labels
                    if other_label in self.sibling_map[anchor_idx]
                ]
                # if there are no suitable sibling positive samples in the current sample, find them from batch_labels
                cross_sample_siblings = []
                if not same_sample_siblings:
                    for sib_label in self.sibling_map[anchor_idx]:
                        if sib_label in label_to_batch:
                            cross_sample_siblings.extend(
                                [(b, sib_label) for b in label_to_batch[sib_label]]
                            )

                valid_sibling_positives = same_sample_siblings + cross_sample_siblings

                # positive sampling
                if valid_sibling_positives:
                    pos_dists = [1 / (distance_matrix_np[anchor_idx][p[1]] + 1e-6) for p in valid_sibling_positives]
                    pos_probs = np.array(pos_dists) / sum(pos_dists)
                    pos_bid, positive_idx = valid_sibling_positives[np.random.choice(len(valid_sibling_positives), p=pos_probs)]

                invalid_negatives = set(current_labels)
                valid_negatives = []
                candidate_negs = []
                for n in label_to_batch:
                    if n in invalid_negatives or n == anchor_idx:
                        continue
                    if self._is_ancestor_or_descendant(anchor_idx, n):
                        continue
                    if positive_idx == None or pos_bid == None:
                        continue
                    # d_T(v_a, v_n) < d_T(v_a, v_p)
                    if distance_matrix_np[anchor_idx][n] < distance_matrix_np[anchor_idx][positive_idx]:
                        candidate_negs.append(n)

                # negative sampling
                if candidate_negs:
                    for neg_label in candidate_negs:
                        valid_negatives.extend(
                            [(b, neg_label) for b in label_to_batch[neg_label]]
                        )

                    neg_bid, negative_idx = valid_negatives[np.random.choice(len(valid_negatives))]

                if positive_idx != None and negative_idx != None:
                    batch_triplets.append((bid, anchor_idx, pos_bid, positive_idx, neg_bid, negative_idx, 'sibling'))

                # 2. parent-child triplets
                # positive sample selection: find parent or child nodes from the current sample
                valid_parent_child_positives = []
                if anchor_idx in self.parent_map:
                    parent_idx = self.parent_map[anchor_idx]
                    if parent_idx in current_labels:
                        valid_parent_child_positives.append((bid, parent_idx))
                if anchor_idx in self.children_map:
                    for child_idx in self.children_map[anchor_idx]:
                        if child_idx in current_labels:
                            valid_parent_child_positives.append((bid, child_idx))

                if valid_parent_child_positives:
                    pos_bid, positive_idx = valid_parent_child_positives[
                        np.random.choice(len(valid_parent_child_positives))]

                invalid_negatives = set(current_labels)
                valid_negatives = []
                candidate_negs = []
                for n in label_to_batch:
                    if n in invalid_negatives or n == anchor_idx:
                        continue
                    if self._is_ancestor_or_descendant(anchor_idx, n):
                        continue
                    if positive_idx == None or pos_bid == None:
                        continue
                    if distance_matrix_np[anchor_idx][n] < distance_matrix_np[anchor_idx][positive_idx]:
                        candidate_negs.append(n)
                if candidate_negs:
                    for neg_label in candidate_negs:
                        valid_negatives.extend(
                            [(b, neg_label) for b in label_to_batch[neg_label]]
                        )
                    neg_bid, negative_idx = valid_negatives[np.random.choice(len(valid_negatives))]

                if positive_idx != None and negative_idx != None:
                    batch_triplets.append((bid, anchor_idx, pos_bid, positive_idx, neg_bid, negative_idx, 'parent_child'))

        return batch_triplets

    def _get_path_to_root(self, node_idx):
        """get the path to the Root"""
        path = []
        current = node_idx
        while current in self.parent_map:
            path.append(current)
            current = self.parent_map[current]
        path.append(self.root_idx)
        return path

    def _find_lca(self, idx1, idx2):
        """find lca"""
        path1 = self._get_path_to_root(idx1)
        path2 = self._get_path_to_root(idx2)
        for node in path1:
            if node in path2:
                return node
        return self.root_idx

    def _is_same_leaf(self, idx1, idx2):
        """determine if the same leaf node"""
        return len(set(self._get_leaves(idx1)) & set(self._get_leaves(idx2))) > 0

    def _get_leaves(self, idx):
        """get leaf nodes"""
        if idx not in self.children_map:
            return [idx]
        leaves = []
        for child in self.children_map[idx]:
            leaves.extend(self._get_leaves(child))
        return leaves

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, num_labels, feature_dim]
            labels: [batch_size, num_labels]
        """
        device = features.device
        self.distance_matrix = self.distance_matrix.to(device)
        triplets = self._get_triplets(labels)
        if not triplets:
            return torch.tensor(0.0, device=device)

        anchors, positives, negatives, triplet_types = [], [], [], []
        for a_bid, a_idx, p_bid, p_idx, n_bid, n_idx, triplet_type in triplets:
            anchors.append(features[a_bid, a_idx])
            positives.append(features[p_bid, p_idx])
            negatives.append(features[n_bid, n_idx])
            triplet_types.append(triplet_type)

        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        def cosine_distance(x, y):
            return 0.5 * (1 - F.cosine_similarity(x, y))

        dist_pos = cosine_distance(anchors, positives)
        dist_neg = cosine_distance(anchors, negatives)

        batch_margins = []
        weights = []
        for a_bid, a_idx, p_bid, p_idx, n_bid, n_idx, triplet_type in triplets:
            d_ap = self.distance_matrix[a_idx][p_idx].item()
            d_an = self.distance_matrix[a_idx][n_idx].item()
            M_t = d_ap - d_an
            margin = self.margin_e + self.alpha * M_t
            batch_margins.append(margin)
            weights.append(1.0 if triplet_type == 'parent_child' else 0.5)

        margins = torch.tensor(batch_margins, device=device)
        weights = torch.tensor(weights, device=device)
        losses = F.relu(dist_pos - dist_neg + margins)
        weighted_losses = losses * weights
        return weighted_losses.mean()

