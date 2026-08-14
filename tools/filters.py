import cv2
import numpy as np
import random
import time

FILTERS = []
def register_filter(func):
    FILTERS.append(func)
    return func

@register_filter
def filter_1(image_rgb: np.ndarray):
    out = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    out[gray < 60] = (15, 8, 10)
    out[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    out[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    out[gray >= 195] = (235, 240, 240)

    return out

@register_filter
def filter_2(image_rgb: np.ndarray):
    out = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % 6) - cell / 2
    cy = (yy % 6) - cell / 2

    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius

    out[dot_mask] = (15, 15, 15)
    return out

@register_filter
def filter_3(image_rgb: np.ndarray):    
    shift = 10
    b, g, r = cv2.split(image_rgb)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)

    out = cv2.merge([b_shift, g, r_shift])
    out[::3, :, :] = (out[::3, :, :] * 5).astype(np.uint8)

    return out

@register_filter
def filter_4(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

@register_filter
def filter_5(image_rgb: np.ndarray):
    h, w = image_rgb.shape[:2]

    sepia_kernel = np.array(
        [
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189],
        ]
    )
    sepia = cv2.transform(image_rgb, sepia_kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = h / 2, w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2) or 1.0
    vignette = np.clip(1 - 0.5 * (dist / max_dist), 0, 1)[..., None]

    out = (sepia * vignette).astype(np.uint8)
    noise = np.random.randint(0, 25, out.shape, dtype=np.uint8)
    return cv2.add(out, noise)

@register_filter
def filter_6(image_rgb: np.ndarray):
    blurred = cv2.GaussianBlur(image_rgb, (35, 35), 0)
    return cv2.addWeighted(blurred, 0.55, image_rgb, 0.45, 0.3)

@register_filter
def filter_7(image_rgb: np.ndarray):
    out = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 5

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.3)
    dot_mask = dist_center < radius

    out[dot_mask] = (55, 20, 130)
    return out

@register_filter
def filter_8(image_rgb: np.ndarray):
    out = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    out[mask == 255] = (10, 140, 255)
    out[mask == 0] = (180, 30, 220)
    return out

@register_filter
def filter_9(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image_rgb, 9, 250, 250)
    return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

@register_filter
def filter_10(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

@register_filter
def filter_11(image_rgb: np.ndarray, block_size: int = 14):
    h, w = image_rgb.shape[:2]
    if h < 2 or w < 2:
        return image_rgb
    small = cv2.resize(image_rgb, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

@register_filter
def filter_12(image_rgb: np.ndarray):
    h, w = image_rgb.shape[:2]
    if h < 2 or w < 2:
        return image_rgb
    b, g, r = cv2.split(image_rgb)
    shift = random.randint(4, 12)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    out = cv2.merge([b, g, r])
    for _ in range(2):
        y = random.randint(0, h - 1)
        out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
    return out

@register_filter
def filter_13(image_rgb: np.ndarray):
    return 255 - image_rgb

@register_filter
def filter_14(image_rgb: np.ndarray):
    b, _, r = cv2.split(image_rgb)
    zeros = np.zeros_like(b)
    return cv2.merge([zeros, zeros, r])

@register_filter
def filter_15(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
    return cv2.bitwise_and(colored, colored, mask=edges)

@register_filter
def filter_16(image_rgb: np.ndarray):
    h, w = image_rgb.shape[:2]
    t = time.time() * 5.0
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
    rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
    return cv2.addWeighted(image_rgb, 0.3, rainbow, 0.7, 0)

@register_filter
def filter_17(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    neon = np.zeros_like(image)

    edges = cv2.Canny(gray, 80, 150)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    neon[:, :, 0] = edges
    neon[:, :, 1] = edges
    return cv2.addWeighted(image,0.3,neon,1.0,0)

@register_filter
def filter_18(image: np.ndarray):
    levels = 6
    step = 256 // levels

    result = (image // step) * step
    return result.astype(np.uint8)

@register_filter
def filter_19(image: np.ndarray):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    result = np.zeros_like(image)

    result[:, :, 0] = gray
    result[:, :, 2] = 255 - gray
    return result

@register_filter
def filter_20(image: np.ndarray):
    kernel = np.array([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2]
    ])

    embossed = cv2.filter2D(image, -1, kernel)
    return embossed