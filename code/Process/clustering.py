
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import math
import numpy as np
import matplotlib.pyplot as plt
from Process.similarity_score import hu_distance_map, chamfer_distance_map, hog_distance_map

def cluster_color_groups(color_groups, distance_fn,
    min_clusters=2, max_clusters=None, linkage_method="average", mask_distance=None):
    color_ids = list(color_groups.keys())
    n_groups = len(color_ids)

    if n_groups == 0:
        raise ValueError("color_groups cannot be empty")

    if any(len(color_groups[cid]) == 0 for cid in color_ids):
        print(f'error here, {[len(color_groups[cid]) for cid in color_ids]}')
        raise ValueError("Every color group must contain at least one mask")

    group_distance = np.zeros((n_groups, n_groups), dtype=float)
    offsets = np.cumsum([0] + [len(color_groups[cid]) for cid in color_ids])

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            if mask_distance is not None:
                distances = mask_distance[offsets[i]:offsets[i + 1], offsets[j]:offsets[j + 1]].ravel()
            else:
                distances = [
                    distance_fn(mask_a, mask_b)
                    for mask_a in color_groups[color_ids[i]]
                    for mask_b in color_groups[color_ids[j]] ]

            d = float(np.mean(distances))
            group_distance[i, j] = d
            group_distance[j, i] = d

    # If only one color group, return a single cluster with all masks.
    if n_groups == 1:
        group_labels = {color_ids[0]: 0}
        mask_labels = { cid: [0] * len(masks) for cid, masks in color_groups.items() }
        return mask_labels, group_labels, group_distance, None
    # two groups then keep them together    
    if n_groups == 2:
        group_labels = {cid: i for i, cid in enumerate(color_ids)}
        mask_labels = { cid: [group_labels[cid]] * len(masks) for cid, masks in color_groups.items()}
        return mask_labels, group_labels, group_distance, None

    # Hierarchical clustering requires a condensed distance matrix.
    condensed_distance = squareform(group_distance, checks=False)

    if max_clusters is None:
        max_clusters = n_groups 
    max_clusters = min(max_clusters, n_groups)
    min_clusters = max(2, min_clusters)

    best_score = -np.inf
    best_labels = None

    for k in range(min_clusters, max_clusters + 1):
        tree = linkage(condensed_distance, method=linkage_method)
        labels = fcluster(tree, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= n_groups:
            continue
        score = silhouette_score(group_distance, labels, metric="precomputed")

        if score > best_score:
            best_score = score
            best_labels = labels
    # use a minimum silhouette score to decide whether to cluster or not; if the best score is too low, keep all groups separate
    # otherwise it might be over-clustered and merge groups that are not really similar
    min_score = 0.3   # tune this; higher = stricter
    if best_labels is None or best_score < min_score:
        best_labels = np.arange(n_groups)   # every group stays separate
        best_score = None
    # assign labels
    unique_labels = sorted(np.unique(best_labels))
    remap = {old: new for new, old in enumerate(unique_labels)}
    best_labels = np.array([remap[x] for x in best_labels])
    group_labels = { cid: int(label)  for cid, label in zip(color_ids, best_labels) }
    mask_labels = {  cid: [group_labels[cid]] * len(masks) for cid, masks in color_groups.items()}
    return mask_labels, group_labels, group_distance, best_score

def plot_mask_clusters(color_groups, mask_labels, n_cols=3):
    # Collect masks by final cluster, retaining their original color-group IDs.
    clusters = {}
    for color_id, masks in color_groups.items():
        labels = mask_labels[color_id]
        for idx, (mask, cluster_id) in enumerate(zip(masks, labels)):
            clusters.setdefault(cluster_id, []).append( (color_id, idx, mask) )

    # Plot each shape cluster separately.
    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]
        n_masks = len(members)
        n_rows = math.ceil(n_masks / n_cols)
        fig, axes = plt.subplots( n_rows, n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            squeeze=False )
        for ax in axes.flat:
            ax.axis("off")
        for ax, (color_id, mask_idx, mask) in zip(axes.flat, members):
            ax.imshow(mask, cmap="gray")
            ax.set_title(f"{color_id} | Mask {mask_idx}")
            ax.axis("off")
        # Show which original color groups contributed to this cluster.
        source_groups = list(dict.fromkeys( color_id for color_id, _, _ in members ))
        fig.suptitle(
            f"Shape Cluster {cluster_id} — {n_masks} masks\n"
            f"Color groups: {', '.join(map(str, source_groups))}",
            fontsize=12)
        plt.tight_layout()
        plt.show()

def build_prototype_library(color_groups, mask_labels, distance_metric):
    # Build a prototype library for each cluster, using the medoid as the prototype.
    # HOG distance is the default choice, can try different distance metrics if needed. 
    all_masks = []
    all_metadata = []

    for color_id, masks in color_groups.items():
        for idx, mask in enumerate(masks):
            all_masks.append(mask)
            all_metadata.append({
                "color_id": color_id,
                "mask_index": idx,
                "cluster_id": mask_labels[color_id][idx]})

    if distance_metric =='hu_moment':
        distance_matrix, _ = hu_distance_map(all_masks)
    elif distance_metric == "chamfer_distance":
        distance_matrix, _ = chamfer_distance_map(all_masks)
    elif distance_metric == "hog":
        distance_matrix, _ = hog_distance_map(all_masks)
    else:
        raise ValueError(f'The selected {distance_metric} does not exist!')
    cluster_ids = sorted(set( item["cluster_id"] for item in all_metadata))

    library = {}
    for cluster_id in cluster_ids:
        indices = [i for i, item in enumerate(all_metadata)
            if item["cluster_id"] == cluster_id]
        # Find the medoid within this cluster.
        within = distance_matrix[np.ix_(indices, indices)]
        medoid_local_idx = int(np.argmin(within.mean(axis=1)))
        prototype_idx = indices[medoid_local_idx]

        library[cluster_id] = {
            "prototype": all_masks[prototype_idx],
            "prototype_metadata": all_metadata[prototype_idx],
            "instances": [
                {"mask": all_masks[i],
                "metadata": all_metadata[i]}
                for i in indices]}
    return library