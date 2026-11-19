# render image from svg to png 
import numpy as np
import pymupdf
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

def render_img(svg_file, long_side=2048, pad=0):
    """render to black and white image, with padding"""
    page = pymupdf.open("svg", open(svg_file, "rb").read())[0]
    zoom_w = long_side / page.rect.width 
    zoom_h= long_side / page.rect.height 
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom_w, zoom_h), alpha=True)
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, 
                                                     pix.width, pix.n)
    return np.pad(a[..., 3] > 127, pad)  


def find_thin_regions(mask, show = True):
    "Find distance to background for each foreground pixel, "
    "using OpenCV distance transform"
    binary = mask.astype(np.uint8)
    # Distance from each foreground pixel to the background
    dist = cv2.distanceTransform( binary, cv2.DIST_L2, 0)
    if show:
        plt.imshow(dist, cmap="magma")
        plt.colorbar(label="Distance to background")
        plt.show()
    return dist

def from_array_to_img(arr, save_path):
    "Convert a boolean array to a black and white PIL image"
    img = Image.fromarray(np.where(arr, 0, 255).astype(np.uint8)).convert("RGB")
    img.save(save_path)
    return img

def from_img_to_url(img):
    "Convert a PIL image to a base64 data URL"
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def from_img_to_mask(img):
    "Convert a PIL image to a mask"
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # Black 255, white  0
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return mask 