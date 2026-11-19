from PIL import Image
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt


def enhance_ink_connection(mask, kernel_size=3, iterations=3, dilation_iteration =3, 
                           if_img = False,
                           show = True):
    if not if_img:
        # convert to binary
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    else:
        binary = (np.asarray(mask) == 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
    # dilation
    dilation = cv2.dilate(binary,kernel,iterations = dilation_iteration)
    # close the holes
    enhanced = cv2.morphologyEx( dilation, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if if_img:
        enhanced = 255 - enhanced # convert back to ink
    if show: 
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(binary, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original mask")
        axes[1].imshow(enhanced, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Closing and Dilation: kernel={kernel_size}, iterations={iterations}" )
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    return enhanced


def detect_contours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE, show = True):
    # convert to binary
    binary = np.asarray(mask).astype(np.uint8)
    binary = (binary > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours( binary.copy(), mode, method)

    # Draw each contour in a different color
    if show:
        # random pick the color
        visualization = np.zeros((*binary.shape, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        colors = rng.integers(60, 256, size=(len(contours), 3), dtype=np.uint8)

        for i, contour in enumerate(contours):
            cv2.drawContours( visualization, contours, i,
                color=tuple(map(int, colors[i])),
                thickness=2)

        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(binary, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original binary mask")
        axes[1].imshow(visualization)
        axes[1].set_title(f"Contours: {len(contours)}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        print(f"Number of contours: {len(contours)}")
        if hierarchy is not None:
            print(f"Hierarchy shape: {hierarchy.shape}")
    return contours, hierarchy


def fill_outer_contours( mask, show= False):
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    filled = np.zeros_like(binary)
    contours, _ = cv2.findContours(
                    binary.copy(),
                    cv2.RETR_EXTERNAL, # different retro function compared to find contour, draw with only the out lines.
                    cv2.CHAIN_APPROX_SIMPLE
                )
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    if show: 
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(binary, cmap="gray")
        axes[0].set_title("Original mask")
        axes[1].imshow(filled, cmap="gray")
        axes[1].set_title("Filled outer contours")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()
    return filled

def fill_closed_contours(mask, show=False, min_area=1):
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    filled = np.zeros_like(binary)

    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE )

    for contour in contours:
        area = cv2.contourArea(contour)
        # Ignore tiny contours
        if area < min_area:
            continue
        # filter by area, only fill the closed contours
        if area > 0:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

    if show:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(binary, cmap="gray")
        axes[0].set_title("Original mask")
        axes[1].imshow(filled, cmap="gray")
        axes[1].set_title("Filled contours")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return filled

def watershed_segments(filled_mask, peak_ratio=0.15, show = True):
    binary = (np.asarray(filled_mask) > 0).astype(np.uint8)
    # Distance 
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Set the threshold for cutting
    #_, sure_fg = cv2.threshold( dist, peak_ratio * dist.max(), 255,cv2.THRESH_BINARY)
    #sure_fg = sure_fg.astype(np.uint8)
    # threshold per blob: a thin unfilled stroke still gets its own seed
    n, blobs = cv2.connectedComponents(binary)
    blob_max = np.zeros(n)
    np.maximum.at(blob_max, blobs.ravel(), dist.ravel())
    sure_fg = (dist > peak_ratio * blob_max[blobs]).astype(np.uint8) * 255

    # Identify background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0
    image = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(image, markers.astype(np.int32))
    # Pick the random color
    rng = np.random.default_rng(42)
    result = np.zeros((*binary.shape, 3), dtype=np.uint8)

    labels = np.unique(markers)
    for label in labels:
        if label <= 1:
            continue
        result[markers == label] = rng.integers( 60, 256, size=3, dtype=np.uint8)

    # Watershed boundaries
    result[markers == -1] = [255, 0, 0]
    if show:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(binary, cmap="gray")
        axes[0].set_title("Filled mask")
        axes[1].imshow(dist, cmap="magma")
        axes[1].set_title("Distance transform")
        axes[2].imshow(result)
        axes[2].set_title("Watershed segments")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    return markers, result

def project_segments_to_original(markers, original_mask, min_area=50):
    original = np.asarray(original_mask) > 0
    candidate_masks = {}
    for label in np.unique(markers):
        if label <= 1:
            continue
        candidate = (markers == label) & original
        # return the mask candidates lists 
        if candidate.sum() >= min_area:
            candidate_masks[int(label)] = candidate.astype(np.uint8)
    return candidate_masks

