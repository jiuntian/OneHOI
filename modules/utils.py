from functools import lru_cache
from logging import Logger
import numpy as np
from PIL import Image

class Printer:
    def __init__(self, print, file="") -> None:
        self.print = print
        if file != "":
            self.file = open(file, "w")
        else:
            self.file = None
        
    def __call__(self, string: str, silent=False) -> None:
        if self.print and not silent:
            print(string)
        self.file.write(string + '\n')
        
    def close_file(self) -> None:
        if self.file:
            self.file.close()
            

@lru_cache(maxsize=100)
def info_once(logger: Logger, msg: str) -> None:
    print(msg)

# get enclosing xyxy (normalized) from mask, mask is shape (H,W)
def get_xyxy_from_mask(mask):
    y_nonzero, x_nonzero = np.nonzero(mask)
    if len(x_nonzero) > 0 and len(y_nonzero) > 0:
        x0, x1 = x_nonzero.min(), x_nonzero.max()
        y0, y1 = y_nonzero.min(), y_nonzero.max()
        bbox = [x0, y0, x1, y1]
        # Normalize bbox to [0, 1]
        bbox = [(v / mask.shape[i % 2]).item() for i, v in enumerate(bbox)]
    return bbox


def overlay_rgba_on_pil(image_pil: Image.Image,
                        visualization_rgba: np.ndarray,
                        opacity: float = 1.0) -> Image.Image:
    """
    image_pil: PIL.Image in RGB or RGBA
    visualization_rgba: np.ndarray of shape (H, W, 4), float32/64 in [0,1]
                        visualization[..., :3] = color; visualization[..., 3] = alpha
    opacity: global multiplier for the visualization alpha (0..1)

    Returns: PIL.Image (RGBA) with overlay applied.
    """
    # Ensure RGBA base
    base = image_pil.convert("RGBA")
    bw, bh = base.size

    viz = visualization_rgba
    assert viz.ndim == 3 and viz.shape[2] == 4, "visualization must be HxWx4"
    vh, vw = viz.shape[:2]

    # Resize visualization if needed
    if (vw, vh) != (bw, bh):
        viz_img = Image.fromarray(np.clip(viz * 255.0, 0, 255).astype(np.uint8))
        viz_img = viz_img.resize((bw, bh), resample=Image.BILINEAR)
        viz = np.asarray(viz_img).astype(np.float32) / 255.0  # back to [0,1] float

    # Separate channels and apply global opacity
    v_rgb = viz[..., :3]
    v_a   = np.clip(viz[..., 3] * float(opacity), 0.0, 1.0)

    # Convert base to float
    base_np = np.asarray(base).astype(np.float32) / 255.0
    b_rgb = base_np[..., :3]
    b_a   = base_np[..., 3]

    # Alpha composite: out = v + b*(1 - v_a)
    out_rgb = v_rgb * v_a[..., None] + b_rgb * (1.0 - v_a[..., None])
    # Keep original base alpha (or set to 1.0 if you prefer opaque)
    out_a = np.clip(b_a + v_a * (1.0 - b_a), 0.0, 1.0)

    out = np.dstack([out_rgb, out_a])
    out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out)