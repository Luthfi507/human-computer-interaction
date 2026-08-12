import cv2
import numpy as np

def filter_1(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    image_rgb[gray < 60] = (15, 8, 10)
    image_rgb[(gray >= 60) & (gray < 130)] = (118, 30, 214)
    image_rgb[(gray >= 130) & (gray < 195)] = (35, 140, 235)
    image_rgb[gray >= 195] = (235, 240, 240)

    return image_rgb

def filter_2(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % 6) - cell / 2
    cy = (yy % 6) - cell / 2

    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius

    image_rgb[dot_mask] = (15, 15, 15)

    return image_rgb

def filter_3(image_rgb: np.ndarray):    
    shift = 10
    b, g, r = cv2.split(image_rgb)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)

    out = cv2.merge([b_shift, g, r_shift])
    image_rgb[::3, :, :] = (out[::3, :, :] * 5).astype(np.uint8)

    return image_rgb

def filter_4(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return image_rgb

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
    image_rgb = cv2.add(out, noise)

    return image_rgb

def filter_6(image_rgb: np.ndarray):
    blurred = cv2.GaussianBlur(image_rgb, (35, 35), 0)
    image_rgb = cv2.addWeighted(blurred, 0.55, image_rgb, 0.45, 0.3)
    return image_rgb

def filter_7(image_rgb: np.ndarray):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 5

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.3)
    dot_mask = dist_center < radius

    out = np.full_like(image_rgb, (215, 190, 245))
    out[dot_mask] = (55, 20, 130)
    return out

FILTERS = [
    filter_1,
    filter_2,
    filter_3,
    filter_4,
    filter_5,
    filter_6,
    filter_7
]