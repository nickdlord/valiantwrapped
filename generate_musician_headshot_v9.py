# ADD THESE IMPORTS AT TOP (keep your existing ones)
import numpy as np
import cv2


# =========================
# SUBJECT-AWARE CROPPING
# =========================

def trim_uniform_borders(img, threshold=18):
    img_np = np.array(img)
    mask = np.any(img_np > threshold, axis=2)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    return img.crop((x0, y0, x1, y1))


def detect_subject_bbox(img):
    """
    Detects main subject using edge density (robust for stylized images)
    """
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect regions
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Largest contour = main subject
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return (x, y, x + w, y + h)


def subject_crop_square(img, zoom=1.18, output_size=1024):
    """
    Crop square around detected subject instead of center
    """
    img = trim_uniform_borders(img)

    bbox = detect_subject_bbox(img)

    w, h = img.size

    if bbox:
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
    else:
        # fallback to center
        cx, cy = w // 2, h // 2

    # determine square crop size
    side = int(min(w, h) / zoom)
    side = max(256, side)

    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)

    right = min(w, left + side)
    bottom = min(h, top + side)

    # adjust if hitting boundaries
    if right - left < side:
        left = max(0, right - side)
    if bottom - top < side:
        top = max(0, bottom - side)

    cropped = img.crop((left, top, right, bottom))

    return cropped.resize((output_size, output_size), Image.Resampling.LANCZOS)


# =========================
# MODIFY YOUR SAVE STEP
# =========================

# FIND THIS PART IN YOUR CODE:
# final_image.save(output_path)

# REPLACE WITH:

final_image = subject_crop_square(
    final_image,
    zoom=args.tile_zoom,
    output_size=args.image_size,
)

final_image.save(output_path)


# =========================
# ADD CLI ARGUMENT
# =========================

# FIND your argparse section and ADD:

parser.add_argument(
    "--tile-zoom",
    type=float,
    default=1.18,
    help="Zoom factor for subject-aware cropping (higher = tighter crop)"
)
