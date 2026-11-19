import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.feature import hog


# ------------------------------------------------------
# normalization and transforms
# ------------------------------------------------------

def normalize_mask(mask, size=256, stroke=3, rms_frac=1/6):
    """
    rescale a mask to a fixed-size square, with a fixed-width stroke, centered on its centroid.
    For later similarity comparisons, make sure the similarity metric is invariant to translation, rotation, and scale."""
    m = np.asarray(mask)
    if m.ndim == 3:  # RGB or BGR input
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    fg = m > 0
    if not fg.any():
        raise ValueError("Mask is empty")

    #skeletonize at original resolution
    ys, xs = np.nonzero(skeletonize(fg))
    if len(xs) == 0:  # single-dot case
        ys, xs = np.nonzero(fg)
    pts = np.column_stack([xs, ys]).astype(float)

    # center on centroid, scale by RMS radius
    c = pts.mean(0)
    r = max(np.sqrt(((pts - c) ** 2).sum(1).mean()), 1e-6)
    pts = (pts - c) * (size * rms_frac / r) + size / 2.0

    #redraw with a fixed stroke width
    out = np.zeros((size, size), np.uint8)
    p = np.round(pts).astype(int)
    keep = (p >= 0).all(1) & (p < size).all(1)
    out[p[keep, 1], p[keep, 0]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke, stroke))
    return cv2.dilate(out, k)


def transform_mask(mask, angle, scale=1.0, mirror=False):
    """rotate or flap for distance calculation"""
    h, w = mask.shape
    src = np.fliplr(mask).copy() if mirror else mask.copy()
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def _norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)

# ------------------------------------------------------
# HOG
# ------------------------------------------------------

def hog_features(mask, norm=True, hog_size=128, cell=8):
    """HOG descriptor of a mask. Set norm=False if the mask is already normalized."""
    m = normalize_mask(mask) if norm else (np.asarray(mask) > 0)
    m = m.astype(np.float32)
    # slight blur so thin lines tolerate small misalignments
    m = cv2.GaussianBlur(m, (5, 5), 0)
    # downsample for speed; INTER_AREA keeps thin lines as soft gray
    if hog_size and m.shape[0] != hog_size:
        m = cv2.resize(m, (hog_size, hog_size), interpolation=cv2.INTER_AREA)
    return hog(m, orientations=9, pixels_per_cell=(cell, cell),
               cells_per_block=(2, 2), feature_vector=True)


def hog_distance(mask_a, mask_b, **hog_kwargs):
    """single distance calculation between two masks"""
    a = _norm(hog_features(mask_a, **hog_kwargs))
    b = _norm(hog_features(mask_b, **hog_kwargs))
    return float(1.0 - a @ b)


def hog_distance_map(candidates_list, angles=range(0, 360, 15), mirror=True,
                     hog_size=128, cell=8):
    """
    calculate the pairwise distance and similarity maps for a list of masks using HOG features.
    note we add the cases of rotation and mirror to the distance calculation.
    """
    masks = [normalize_mask(m) for m in candidates_list]
    if len(masks) == 0:
        raise ValueError("candidates_list is empty")
    flips = [False, True] if mirror else [False]
    feat = lambda m: hog_features(m, norm=False, hog_size=hog_size, cell=cell)
    ref = _norm(np.stack([feat(m) for m in masks]))
    # each mask under a range of rotation angles 
    bank = _norm(np.stack([ np.stack([feat(transform_mask(m, a, 1.0, f)) for f in flips for a in angles])
        for m in masks ]))

    # best cosine similarity between mask i and any transform of mask j
    cos = np.einsum("id,jtd->ijt", ref, bank).max(axis=2)
    distance_map = 1.0 - cos
    distance_map = 0.5 * (distance_map + distance_map.T)   # make symmetric
    np.fill_diagonal(distance_map, 0.0)
    similarity_map = np.clip(1.0 - distance_map, 0.0, 1.0)
    return distance_map, similarity_map


# ------------------------------------------------------
# chamfer distance
# ------------------------------------------------------

def _ds(mask):
    """Distance from every pixel to the nearest line pixel."""
    return cv2.distanceTransform((mask == 0).astype(np.uint8),
                                 cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def chamfer_distance(a, b, da=None, ref_len=None):
    """
    pairwise chamfer distance between two masks, normalized by the RMS radius of a. (size/6 with normalize_mask defaults)
    instead of the canvas diagonal, so values are in "fractions of the shape
    size" and similarities spread out. 
    """
    ea, eb = a > 0, b > 0
    if not ea.any() or not eb.any():
        return 1.0
    da = _ds(a) if da is None else da
    db = _ds(b)
    ref_len = ref_len or a.shape[0] / 6
    return float((db[ea].mean() + da[eb].mean()) / (2 * ref_len))


def best_transform_choice(a, b, angles, scales, allow_mirror,
                          refine_step, refine_span, ref_len=None):
    da = _ds(a)   # only compute once for efficiency
    dist = lambda ang, sc, mir: chamfer_distance(
        a, transform_mask(b, ang, sc, mir), da=da, ref_len=ref_len)

    best_distance, best_transform = float("inf"), None
    for mirror in ([False, True] if allow_mirror else [False]):
        for scale in scales:
            for angle in angles:
                d = dist(angle, scale, mirror)
                if d < best_distance:
                    best_distance = d
                    best_transform = {"angle": float(angle), "scale": scale, "mirror": mirror}

    if refine_step and refine_span:
        t = best_transform
        for angle in np.arange(t["angle"] - refine_span,
                               t["angle"] + refine_span + 1e-9, refine_step):
            d = dist(angle, t["scale"], t["mirror"])
            if d < best_distance:
                best_distance = d
                best_transform = {**t, "angle": float(angle % 360)}
    return best_distance, best_transform


def shape_similarity(mask_a, mask_b, angles=range(0, 360, 10), scales=(0.9, 1.0, 1.1),
                     allow_mirror=True, refine_step=2, refine_span=8,
                     sharpness=5.0, normalized=False):
    """Best chamfer similarity."""
    a = mask_a if normalized else normalize_mask(mask_a)
    b = mask_b if normalized else normalize_mask(mask_b)
    best_distance, best_transform = best_transform_choice(
        a, b, angles, scales, allow_mirror, refine_step, refine_span)
    return {"similarity": float(np.exp(-sharpness * best_distance)),
            "distance": best_distance, **best_transform}


def chamfer_distance_map(candidates_list, sharpness=5.0, **kwargs):
    masks = [normalize_mask(m) for m in candidates_list]   # normalize once
    n = len(masks)
    distance_map = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            res = shape_similarity(masks[i], masks[j], sharpness=sharpness,
                                   normalized=True, **kwargs)
            distance_map[i, j] = distance_map[j, i] = res["distance"]
    similarity_map = np.exp(-sharpness * distance_map)
    return distance_map, similarity_map


# ------------------------------------------------------
# Hu moments
# ------------------------------------------------------

def hu_moments(mask, eps=1e-12, norm=True, mirror=False, n_terms=7):
    """Log-scaled Hu moments. keep only the first n_terms (default 7) for similarity calculation. Set norm=False if the mask is already normalized."""
    if norm:
        mask = normalize_mask(mask)
    binary = (mask > 0).astype(np.uint8)
    if mirror:
        binary = cv2.flip(binary, 1)
    hu = cv2.HuMoments(cv2.moments(binary, binaryImage=True)).flatten()[:n_terms]
    return -np.sign(hu) * np.log10(np.abs(hu) + eps)
    # h7 changes sign under mirroring; with n_terms < 7 mirror has no effect.


def hu_distance(a, b, mirror=True, norm=True, n_terms=7):
    ma = hu_moments(a, norm=norm, n_terms=n_terms)
    distance = np.linalg.norm(ma - hu_moments(b, norm=norm, n_terms=n_terms))
    if mirror:
        mb = hu_moments(b, mirror=True, norm=norm, n_terms=n_terms)
        distance = min(distance, np.linalg.norm(ma - mb))
    return float(distance)


def hu_distance_map(candidates_list, tau=2.0, mirror=True, norm=True,
                    n_terms=7, standardize=False):
    moments_array = np.array([hu_moments(m, norm=norm, n_terms=n_terms)
                              for m in candidates_list])
    if standardize:
        mu = moments_array.mean(0)
        sd = moments_array.std(0) + 1e-9
        moments_array = (moments_array - mu) / sd

    distance_map = cdist(moments_array, moments_array, metric="euclidean")
    if mirror:
        mirror_array = np.array([hu_moments(m, mirror=True, norm=norm, n_terms=n_terms)
                                 for m in candidates_list])
        if standardize:
            mirror_array = (mirror_array - mu) / sd
        distance_map = np.minimum(distance_map,
                                  cdist(moments_array, mirror_array, metric="euclidean"))

    similarity_map = np.exp(-distance_map / tau)
    return distance_map, similarity_map


# ------------------------------------------------------
# metrics
# ------------------------------------------------------

METRICS = {
    "hu_moment": hu_distance_map,
    "chamfer_distance": chamfer_distance_map,
    "hog": hog_distance_map,
}

def similarity_map_for(candidates_list, metric, **kwargs):
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {list(METRICS)}")
    return METRICS[metric](candidates_list, **kwargs)


# ------------------------------------------------------
# scoring
# ------------------------------------------------------

def score_candidates(candidates_list, lambda_penalty=0.1, alpha=0.8,
                     score_metric="hu_moment", weights=None, combine_mode="mean",
                     robust="spread", **metric_kwargs):

    if isinstance(score_metric, str):
        _, similarity_map = similarity_map_for(candidates_list, score_metric, **metric_kwargs)
    else:
        raise ValueError("Multiple metrics not yet implemented in score_candidates")
        # before support combined, but hard to tune weight, delete that part for now.
    
    similarities = np.asarray(similarity_map, dtype=float)
    n = similarities.shape[0]
    if n == 0:
        raise ValueError("No similarity is calculated.")
    if (not np.all(np.isfinite(similarities))
            or np.any(similarities < 0) or np.any(similarities > 1)):
        raise ValueError(f"Similarities must be in [0, 1], got "
                         f"max={similarities.max()}, min={similarities.min()}")

    pairwise = similarities[np.triu_indices(n, k=1)]
    if pairwise.size == 0:
        q_avg = q_robust = 0.0
    else:
        q_avg = pairwise.mean()
        if robust == "spread":
            q_robust = 1.0 - (pairwise.max() - pairwise.min())
        elif robust == "min":
            q_robust = pairwise.min()
        else:
            raise ValueError(f"Unknown robust {robust!r}; use 'spread' or 'min'")

    quality = alpha * q_avg + (1 - alpha) * q_robust
    if pairwise.size == 0:
        quality = 0.6 # set as threshold to one piece
    complexity_penalty = lambda_penalty * np.log(n)   # log(1) = 0 for one piece
    
    return float(quality - complexity_penalty)