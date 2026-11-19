from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from skimage.color import rgb2lab, lab2rgb, deltaE_ciede2000
import pandas as pd
from sklearn.metrics import silhouette_score
from skimage.restoration import denoise_bilateral
from scipy.spatial.distance import cdist

#---------------------------------------------------------------
# Kmeans quantizing
#-------------------------------------------------------------
# This arrarys follow the original blog post: https://bottosson.github.io/posts/oklab/
_RGB_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                     [0.2119034982, 0.6806995451, 0.1073969566],
                     [0.0883024619, 0.2817188376, 0.6299787005]])
_LMS_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]])
_LAB_LMS = np.linalg.inv(_LMS_LAB)
_LMS_RGB = np.linalg.inv(_RGB_LMS)
# convert to oklab color
def to_oklab(rgb):
    """uint8 RGB (..., 3) -> Oklab x 100, L in 0-100."""
    c = np.asarray(rgb, np.float32) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)    # undo sRGB gamma
    return (100 * (np.cbrt(c @ _RGB_LMS.T) @ _LMS_LAB.T)).astype(np.float32)

def from_oklab(lab):
    """Oklab x 100 (..., 3) -> uint8 RGB."""
    c = ((np.asarray(lab, np.float64) / 100) @ _LAB_LMS.T) ** 3 @ _LMS_RGB.T
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.abs(c) ** (1 / 2.4) - 0.055)
    return (c.clip(0, 1) * 255).round().astype(np.uint8)

def normalize(values):
    values = np.asarray(values, dtype=float)
    vmin = np.min(values)
    vmax = np.max(values)
    if vmax - vmin < 1e-12:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)

def merge_close_colors(palette, labels, merge_de=8):
    """
    Merge the closest pair of centres while they are closer than merge_de (Oklab x 100),
    weighted by pixel count. 
    The agent color palette is specifically selected to be perceptually distinct. Also 
    because the image returned very often has a few shades of the same color, we merge them to get a more compact palette.
    """
    palette = list(palette)
    weights = list(np.bincount(labels.ravel(), minlength=len(palette)).astype(float))
    groups = [[i] for i in range(len(palette))]   # original labels in each merged colour
    while len(palette) > 1:
        d = cdist(palette, palette)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] >= merge_de:
            break
        w = weights[i] + weights[j]
        palette[i] = (palette[i] * weights[i] + palette[j] * weights[j]) / w
        weights[i] = w
        groups[i] += groups[j]
        del palette[j], weights[j], groups[j]
    lut = np.empty(sum(len(g) for g in groups), dtype=int)
    for new, old in enumerate(groups):
        lut[old] = new
    return np.array(palette), lut[labels]

def quantifize_color(
    img,
    k_range=range(2, 9),
    random_state=42,
    max_silhouette_samples=3000,
    mean_weight=0.65,
    p95_weight=0.25,
    complexity_weight=0.10,
    transform=True,
    denoise=True,
    sigma_color=3,
    sigma_spatial=3,
    round_decimals=3,
    merge_de=8,     # merge colours closer than this; shades of one colour <= 6, distinct colours >= 9.9
):

    img = np.asarray(img, dtype=np.float32)
    if transform:
        original_oklab = to_oklab(img)
    else:
        original_oklab = img.copy()
    h, w = original_oklab.shape[:2]
    original_pixels = original_oklab.reshape(-1, 3)

    if denoise:
        filtered_oklab = denoise_bilateral( original_oklab,
                                            sigma_color=sigma_color,
                                            sigma_spatial=sigma_spatial,
                                            channel_axis=-1)
    else:
        filtered_oklab = original_oklab.copy()
    pixels = filtered_oklab.reshape(-1, 3)
    compressed_pixels = np.round(pixels, decimals=round_decimals)
    unique_colors, inverse, counts = np.unique( compressed_pixels,
                                                axis=0,
                                                return_inverse=True,
                                                return_counts=True)

    # Weighted k-means
    results = []
    models = {}
    rng = np.random.default_rng(random_state)
    # Use the same sample for every K
    n_samples = min( max_silhouette_samples, len(pixels) )
    sample_indices = rng.choice( len(pixels), size=n_samples, replace=False)

    for k in k_range:
        if k > len(unique_colors):
            continue
        model = KMeans(n_clusters=k,
                        random_state=random_state,
                        n_init=10)
        model.fit(unique_colors, sample_weight=counts)

        # Assign all pixels
        labels = model.labels_[inverse]
        palette = model.cluster_centers_
        reconstructed = palette[labels]

        # oklab perceptual reconstruction errors 
        # Euclidean distance in oklab
        # first keep them all to take as reference
        errors_filtered = np.linalg.norm( pixels - reconstructed, axis=1 )
        errors_original = np.linalg.norm( original_pixels - reconstructed, axis=1)
        mean_de = errors_filtered.mean()
        p95_de = np.percentile(errors_filtered, 95)
        mean_original_de = errors_original.mean()
        p95_original_de = np.percentile( errors_original, 95)

        # Silhouette score
        sample_pixels = pixels[sample_indices]
        sample_labels = labels[sample_indices]

        if len(np.unique(sample_labels)) > 1:
            silhouette = silhouette_score( sample_pixels, sample_labels )
        else:
            silhouette = np.nan
        results.append({
            "k": k,
            "mean_deltaE_ok": mean_de,
            "p95_deltaE_ok": p95_de,
            "mean_original_deltaE_ok": mean_original_de,
            "p95_original_deltaE_ok": p95_original_de,
            "inertia": model.inertia_,
            "silhouette": silhouette})
        models[k] = model
    if not results:
        raise ValueError( "No valid K values. Check k_range and image colors.")

    # select the best k
    scores = pd.DataFrame(results)
    norm_mean = normalize(scores["mean_deltaE_ok"])
    norm_p95 = normalize(scores["p95_deltaE_ok"])
    norm_k = normalize(scores["k"])

    scores["selection_score"] = ( mean_weight * norm_mean
        + p95_weight * norm_p95  + complexity_weight * norm_k)
    best_index = scores["selection_score"].idxmin()
    best_k = int(scores.loc[best_index, "k"])

    # reconstruct best quantized image
    best_model = models[best_k]
    best_labels = best_model.labels_[inverse]
    # merge shade color
    best_palette, best_labels = merge_close_colors(best_model.cluster_centers_,
                                                   best_labels, merge_de)
    label_map = best_labels.reshape(h, w)
    quantized_oklab = best_palette[label_map]

    # Convert to rgb for visualization
    quantized_rgb = from_oklab(quantized_oklab)
    filtered_rgb = from_oklab(filtered_oklab)
    return {
        "filtered_oklab": filtered_oklab,
        "filtered_rgb": filtered_rgb,
        "quantized_oklab": quantized_oklab,
        "quantized_rgb": quantized_rgb,
        "labels": label_map,
        "palette_oklab": best_palette,
        "best_k": best_k,  # k chosen by the score, before merging
        "n_colors": len(best_palette),  # colours after merging, white included
        "scores": scores,
        "models": models}, quantized_rgb


def get_color_masked_img(color_img, mask_img, out_path, 
                         return_img = False, if_img = False):
    mask_img = mask_img.convert("L")
    assert color_img.size == mask_img.size, "Image sizes must match"
    # Convert the white part into a mask
    mask_array = np.array(mask_img)
    if if_img:
        mask = mask_array > 128
    else:
        mask = mask_array < 128
    color_array = np.array(color_img)
    color_array[~mask, :] = [255, 255, 255] # Make unselected pixels transparent
    # Save the result
    result = Image.fromarray(color_array)
    result.save(out_path)
    if return_img:
        return result

def get_color_set(img):
    img = np.array(img)
    pixels = img.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    return colors, counts 

def extract_ink_masks( original_path, colored_path, output_path, colors, counts,
    min_area=20, black_threshold=100):
    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    colored = cv2.imread(str(colored_path), cv2.IMREAD_COLOR)

    if original is None or colored is None:
        raise FileNotFoundError("Could not load one or both images.")
    if original.shape[:2] != colored.shape[:2]:
        raise ValueError("Original and recolored images must have identical dimensions.")
    # Identify black ink in the original image.
    ink = np.all(original < black_threshold, axis=2).astype(np.uint8) * 255
    results = []
    for color_index, (color, count) in enumerate(zip(colors, counts), start=1):
        if color.min() >= 250:
            continue # drop white
        # cv2 use BGR, convert rgb first.
        color_bgr = color[::-1]
        # Exact color mask.
        color_mask = np.all(colored == color_bgr, axis=2).astype(np.uint8) * 255
        # Find disconnected components of this color.
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # filter components
        valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
        if not valid_contours:
            continue
        extracted = cv2.bitwise_and(ink, color_mask)
        # drop speckles, noise, and small components
        _, blobs, stats, _ = cv2.connectedComponentsWithStats(extracted, connectivity=8)
        extracted[stats[blobs, cv2.CC_STAT_AREA] < min_area] = 0
        if cv2.countNonZero(extracted) == 0:
            continue
        # Save one combined mask per color.
        if output_path is not None:
            prefix = f"color_{color_index:03d}"
            mask_path = output_path + f"{prefix}_mask.png"
            cv2.imwrite(str(mask_path), extracted)

        results.append({
            "color_id": color_index,
            "color_bgr": color_bgr.tolist(),
            "pixel_count": int(count),
            "component_count": len(valid_contours),
            "mask_path": str(mask_path)
        })

    print(f"Saved {len(results)} color group masks.")
    return results
