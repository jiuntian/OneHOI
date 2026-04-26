import torch
from PIL import Image
import numpy as np
from diffusers.utils import load_image

from pipelines.onehoi import OneHOIPipeline
from modules.utils import get_xyxy_from_mask, overlay_rgba_on_pil

negative_prompt = "worst quality, low quality, monochrome, zombie, interlocked fingers, Aissist, cleavage, nsfw, blur,"
cfg_scale = 3.5
device = "cuda:3"

if __name__ == "__main__":
    
    # Loading the pipeline and moving it to GPU
    pipeline = OneHOIPipeline.from_checkpoint(
        base_model="black-forest-labs/FLUX.1-Kontext-dev",
        checkpoint_path="models/OneHOI",
        torch_dtype=torch.bfloat16,
    )
    pipeline.to(device)

    # Optimize inference speed (optional)
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    pipeline.vae.to(memory_format=torch.channels_last)

    # Prepare inputs
    # visualize masks with different colors (up to 6 masks and colors)
    # the masks is RGBA, the active part is black, the rest is transparent
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path

    # read masks from saved PNGs
    directory = Path("./docs/res/workflow_1/")
    files = [
        "person1.png",
        "dog.png",
        "person2.png",
        "bench.png"
        ]
    imgs = [Image.open(directory / name).convert("RGBA") for name in files]
    arrs = [np.array(img) for img in imgs]

    # active area: black RGB and alpha > 0
    masks = []
    for arr in arrs:
        mask = np.all(arr[..., :3] == 0, axis=-1) & (arr[..., 3] > 0)
        masks.append(mask)
    masks = np.stack(masks, axis=0)

    mix_boxes = [
        [ # b=0
            [ # n=0
                get_xyxy_from_mask(masks[0]), get_xyxy_from_mask(masks[1]),
            ],
            [ # n=1
                get_xyxy_from_mask(masks[2]), get_xyxy_from_mask(masks[3]),
            ],
        ]
    ]
    # mix_box_labels: [B, N, M]
    mix_box_labels = [
        [ # b=0
            ["person", "dog"], # n=0
            ["person", "bench"],
        ]
    ]
    # mix_box_labels: [B, N]
    mix_hoi_labels = [
        [ # b=0
            "hug",
            "lie on",
        ]
    ]
    arbitrary_masks = [ # b=0
            [masks[0], masks[1]], # n=0
            [masks[2], masks[3]],
        ],  # batch size
    from copy import deepcopy
    
    image = load_image("./docs/res/workflow_1/generated_girl.png")
    prompt = f"the person is now hug the dog, while another person is now lie on the bench."
    generator = torch.Generator(device="cpu").manual_seed(2026)
    generated_img = pipeline(
        prompt=prompt,
        image=image,
        # If there are old HOI triplet like InteractDiffusion, 
        #       use boxes, box_labels and hoi_labels
        # Else if contains arbitrary masks, 
        #       use arbitrary_masks.
        # or if contains mixed inputs,
        #       use mix_boxes, mix_box_labels and mix_hoi_labels.
        # boxes= [[s1x1,s1y2,s1x2,s1y2], [o1x1,o1y2,o1x2,o1y2],
        #         [s2x1,s2y2,s2x2,s2y2], [o2x1,o2y2,o2x2,o2y2]
        #        ],  # [B, N, 4]
        # box_labels=box_labels,
        # hoi_labels=hoi_labels,
        mix_boxes=deepcopy(mix_boxes),
        mix_box_labels=deepcopy(mix_box_labels),
        mix_hoi_labels=deepcopy(mix_hoi_labels),
        arbitrary_masks=arbitrary_masks,
        generator=generator,
        negative_prompt=negative_prompt,
        true_cfg_scale=3.5,
        # height=512,
        # width=512,
        # max_area=512**2,
        hoi_seq_len=512,
    ).images[0]
    generated_img.save("sample.jpg")
    
    # visualize the masks on the generated image
    colors = np.array([[1, 0, 0], [0, 0.6, 0], [0.0, 0.7, 0.5], [0.5, 0, 0.8],
                       [0.8, 0.8, 0], [0.0, 0.7, 0.5], [0, 0.8, 0.8]])  # RGB colors for each mask
    height, width = masks.shape[1], masks.shape[2]
    visualization = np.zeros((height, width, 4), dtype=float)
    ## alpha = 1 where any mask is present, 0 otherwise (keeps blank areas transparent)
    visualization[..., 3] = np.any(masks, axis=0).astype(float)
    for i in range(masks.shape[0]):
        for c in range(3):
            visualization[:, :, c] += masks[i] * colors[i, c]
    visualization = np.clip(visualization, 0, 1)
    out_img = overlay_rgba_on_pil(generated_img, visualization, opacity=0.4)
    ## out_img is RGBA; convert to RGB if you want
    out_img_rgb = out_img.convert("RGBA")
    out_img_rgb.save("visualization.png")
    