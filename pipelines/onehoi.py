
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable
import warnings
from typing_extensions import Self

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel, 
    T5EncoderModel,
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPVisionModelWithProjection   
)

from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    EXAMPLE_DOC_STRING,
    calculate_shift,
    retrieve_timesteps,
    PREFERRED_KONTEXT_RESOLUTIONS
)
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import FluxTransformer2DModel
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.utils import (
    logging,
    replace_example_docstring,
    is_torch_xla_available,
)
from safetensors.torch import load_model
from safetensors import safe_open

from modules.grounding_encoder import GroundingEncoder, GroundingInput
from modules.transformers import OneHOITransformer2DModel
from modules.attention_processor import OneHOIAttnProcessor2_0
from modules.utils import info_once

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
    
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class OneHOIPipeline(FluxKontextPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: OneHOITransformer2DModel,
        grounding_encoder: GroundingEncoder,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super(FluxKontextPipeline).__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            grounding_encoder=grounding_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128
        self.transformer.set_attn_processor(OneHOIAttnProcessor2_0())
        
    @classmethod
    def from_checkpoint(cls, base_model: str, checkpoint_path: str, **kwargs) -> Self:
        """
        Load the pipeline from a pretrained model and LoRA checkpoint.
        This only requires to load extra LoRA and grounding encoder weights.
        """
        logger.info(f"Loading transformer from {base_model}")
        transformer = OneHOITransformer2DModel.from_pretrained(base_model, subfolder="transformer", **kwargs)
        
        logger.info(f"Loading grounding encoder from {checkpoint_path}")
        grounding_encoder_config = GroundingEncoder.load_config(Path(checkpoint_path) / "grounding_encoder")
        grounding_encoder = GroundingEncoder.from_config(grounding_encoder_config)
        grounding_encoder.to(dtype=transformer.dtype, device=transformer.device)
        try:
            load_model(grounding_encoder, Path(checkpoint_path) / "grounding_encoder.safetensors")
            print("Loaded grounding encoder weights.")
        except Exception as e:
            print(f"Failed to load grounding encoder: {e} \n Trying to remove module. and load state_dict instead.")
            # Load state dict and remove 'module.' prefix if present
            state_dict = safe_open(Path(checkpoint_path) / "grounding_encoder.safetensors", framework="pt", device='cpu')
            new_state_dict = {}
            with state_dict as f:
                for k in f.keys():
                    v = f.get_tensor(k)
                    new_key = k.replace("module.", "") if k.startswith("module.") else k
                    new_state_dict[new_key] = v
            grounding_encoder.load_state_dict(new_state_dict)
        
        logger.info(f"Loading the rest from {base_model}")
        pipeline = cls.from_pretrained(base_model,
                             grounding_encoder=grounding_encoder,
                             transformer=transformer,
                             **kwargs)
        
        logger.info(f"Loading LoRA weights from {checkpoint_path}")
        pipeline.load_lora_weights(checkpoint_path)
        
        return pipeline

    def check_inputs(self,
                     prompt,
                     prompt_2, 
                     height,
                     width, 
                     boxes=None,
                     box_labels=None,
                     hoi_labels=None,
                     negative_prompt=None,
                     negative_prompt_2=None,
                     prompt_embeds=None,
                     negative_prompt_embeds=None,
                     pooled_prompt_embeds=None,
                     negative_pooled_prompt_embeds=None,
                     callback_on_step_end_tensor_inputs=None,
                     max_sequence_length=None,
                     num_images_per_prompt=1):
        super().check_inputs(prompt,
                             prompt_2,
                             height,
                             width,
                             negative_prompt,
                             negative_prompt_2,
                             prompt_embeds,
                             negative_prompt_embeds,
                             pooled_prompt_embeds,
                             negative_pooled_prompt_embeds,
                             callback_on_step_end_tensor_inputs,
                             max_sequence_length)
        
        if num_images_per_prompt > 1:
            raise ValueError("num_images_per_prompt > 1 is not supported in this pipeline yet.")
        
        if boxes is not None:
            if isinstance(boxes, list) and len(boxes) == 0:
                raise ValueError("Boxes cannot be an empty list.")
            # Single prompt: boxes = [[x1, y1, x2, y2], ...]
            if isinstance(boxes, list) and all(isinstance(b, list) and len(b) == 4 for b in boxes):
                pass
            # Multiple prompts: boxes = [[[x1, y1, x2, y2], ...], ...]
            elif isinstance(boxes, list) and all(isinstance(b, list) and all(isinstance(bb, list) and len(bb) == 4 for bb in b) for b in boxes):
                pass
            else:
                raise ValueError("Boxes must be a list of [x1, y1, x2, y2] or a list of lists of [x1, y1, x2, y2].")
        if box_labels is not None:
            if isinstance(box_labels, list) and len(box_labels) == 0:
                raise ValueError("box_labels cannot be an empty list.")
            # Single prompt: box_labels = ["box1", "box2", ...]
            if isinstance(box_labels, list) and all(isinstance(b, str) for b in box_labels):
                if not isinstance(prompt, str):
                    raise ValueError("When specifying single hoi_labels, prompt must be a single string for single prompt.")
                if boxes is not None and isinstance(boxes, list) and all(isinstance(b, list) and len(b) == 4 for b in boxes):
                    if len(box_labels) != len(boxes):
                        raise ValueError("Number of box_labels must match number of boxes for single prompt.")
            # Multiple prompts: box_labels = [["box1", "box2", ...], ...]
            elif isinstance(box_labels, list) and all(isinstance(b, list) and all(isinstance(bb, str) for bb in b) for b in box_labels):
                if isinstance(prompt, str):
                    raise ValueError("When specifying multi hoi_labels, prompt must be a list of strings for multi prompt.")
                if boxes is not None and isinstance(boxes, list) and all(isinstance(b, list) and all(isinstance(bb, list) and len(bb) == 4 for bb in b) for b in boxes):
                    if len(box_labels) != len(boxes):
                        raise ValueError("Number of box_labels batches must match number of boxes batches for multi-prompt.")
                    for labels, box_batch in zip(box_labels, boxes):
                        if len(labels) != len(box_batch):
                            raise ValueError("Number of box_labels in each batch must match number of boxes in each batch.")
            else:
                raise ValueError("box_labels must be a list of strings or a list of lists of strings.")
            
        if hoi_labels is not None:
            if isinstance(hoi_labels, list) and len(hoi_labels) == 0:
                raise ValueError("hoi_labels cannot be an empty list.")
            # Single prompt: hoi_labels = ["hoi1", "hoi2", ...]
            if isinstance(hoi_labels, list) and all(isinstance(h, str) for h in hoi_labels):
                if not isinstance(prompt, str):
                    raise ValueError("When specifying single hoi_labels, prompt must be a single string for single prompt.")
                if boxes is not None and isinstance(boxes, list) and all(isinstance(b, list) and len(b) == 4 for b in boxes):
                    if len(hoi_labels) != len(boxes) // 2:
                        raise ValueError("Number of hoi_labels must be half the number of boxes for single prompt.")
                    if len(boxes) % 2 != 0:
                        raise ValueError("When specified hoi_labels, number of boxes must be divisible by 2 for single prompt.")
            # Multi prompt: hoi_labels = [["hoi1", "hoi2", ...], ...]
            elif isinstance(hoi_labels, list) and all(isinstance(h, list) and all(isinstance(hh, str) for hh in h) for h in hoi_labels):
                if isinstance(prompt, str):
                    raise ValueError("When specifying multi hoi_labels, prompt must be a list of strings for multi prompt.")
                if boxes is not None and isinstance(boxes, list) and all(isinstance(b, list) and all(isinstance(bb, list) and len(bb) == 4 for bb in b) for b in boxes):
                    if len(hoi_labels) != len(boxes):
                        raise ValueError("Number of hoi_labels batches must match number of boxes batches for multi-prompt.")
                    for hoi_batch, box_batch in zip(hoi_labels, boxes):
                        if len(hoi_batch) != len(box_batch) // 2:
                            raise ValueError("Number of hoi_labels in each batch must be half the number of boxes in each batch for multi-prompt.")
                        if len(box_batch) % 2 != 0:
                            raise ValueError("When specified hoi_labels, number of boxes in each batch must be divisible by 2 for multi-prompt.")
            else:
                raise ValueError("hoi_labels must be a list of strings or a list of lists of strings.")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        boxes: Optional[Union[List[List[float]], List[List[List[float]]]]] = None,
        box_labels: Optional[Union[List[str], List[List[str]]]] = None,
        hoi_labels: Optional[Union[List[str], List[List[str]]]] = None,
        mix_boxes: Optional[Union[List[List[List[float]]], List[List[List[None]]]]] = None,
        mix_box_labels: Optional[Union[List[List[List[str]]]]] = None,
        mix_hoi_labels: Optional[Union[List[List[str]], List[List[None]]]] = None,
        arbitrary_masks: Optional[Union[List[List[torch.Tensor]], List[List[torch.Tensor]]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        hoi_seq_len: int = 512,
        max_box: Optional[int] = None,
        max_hoi: Optional[int] = None,
        use_union_action_mask: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            boxes (`List[List[float]]` or `List[List[List[float]]]`, *optional*):
                Bounding boxes for grounding. For a single image, provide a list of boxes (each box as `[x1, y1, x2, y2]`).
                For a batch, provide a list of lists, where each inner list contains the boxes for one image.
            box_labels (`List[str]` or `List[List[str]]`, *optional*):
                The labels for the bounding boxes.
            hoi_labels (`List[str]` or `List[List[str]]`, *optional*):
                The labels for the human-object interactions. For a single image, provide a list of hoi labels.
                For a batch, provide a list of lists, where each inner list contains the hoi labels for one image.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512):
                Maximum sequence length to use with the `prompt`.
            max_area (`int`, defaults to `1024 ** 2`):
                The maximum area of the generated image in pixels. The height and width will be adjusted to fit this
                area while maintaining the aspect ratio.
            _auto_resize (`bool`, *optional*, defaults to `True`):
                Whether to automatically resize the image to fit the specified dimensions.
            max_hoi (`int`, *optional*, defaults to `4`):
                The maximum number of HOI (Human-Object Interaction) instances to consider.
            max_box (`int`, *optional*, defaults to `12`):
                The maximum number of bounding boxes to consider.
            use_union_action_mask (`bool`, *optional*, defaults to `True`):
                Whether to use the union action mask.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        if height != original_height or width != original_width:
            logger.warning(
                f"Generation `height` and `width` have been adjusted to {height} and {width} to fit the model requirements."
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            boxes=boxes,
            box_labels=box_labels,
            hoi_labels=hoi_labels,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        if boxes is not None and isinstance(boxes, list):
            # If boxes is a single set of coordinates, wrap it in a list
            if isinstance(boxes[0], list) and len(boxes[0]) == 4:
                boxes = [boxes]
        
        if box_labels is not None and isinstance(box_labels, list):
            # If box_labels is a single set of labels, wrap it in a list
            if isinstance(box_labels[0], str):
                box_labels = [box_labels]
        
        if hoi_labels is not None and isinstance(hoi_labels, list):
            # If hoi_labels is a single set of labels, wrap it in a list
            if isinstance(hoi_labels[0], str):
                hoi_labels = [hoi_labels]
                
        # wrap in batch, and check input validity
        if arbitrary_masks is not None and isinstance(arbitrary_masks, list):
            # If arbitrary_masks is a single set of masks, wrap it in a list
            if isinstance(arbitrary_masks[0], torch.Tensor):
                arbitrary_masks = [arbitrary_masks]
            if isinstance(arbitrary_masks[0], list):
                arbitrary_masks = [arbitrary_masks]
            if len(arbitrary_masks) != batch_size:
                raise ValueError(f"Length of arbitrary_masks {len(arbitrary_masks)} must match batch_size {batch_size}.")

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            
        if boxes is not None or box_labels is not None or hoi_labels is not None:
            if mix_box_labels is not None or mix_hoi_labels is not None or mix_boxes is not None:
                raise NotImplementedError("Choose either (boxes, box_labels, hoi_labels) or (mix_boxes, mix_box_labels, mix_hoi_labels). Not both.")
            
        # 2.1 Prepare text embeddings for grounding
        _max_hoi, _max_box = max_hoi, max_box
        if hoi_labels is not None:
            _hoi_seq_len, _max_hoi = GroundingInput.get_hoi_seq_len(hoi_seq_len, max(len(h) for h in hoi_labels))
        elif box_labels is not None:
            _hoi_seq_len, _max_box = GroundingInput.get_box_seq_len(hoi_seq_len, max(len(h) for h in box_labels))
        elif mix_hoi_labels is not None:
            _hoi_seq_len, _max_hoi = GroundingInput.get_hoi_seq_len(hoi_seq_len, max(len(h) for h in mix_hoi_labels))
            _max_box = _max_hoi * 3
        else:
            _hoi_seq_len, _max_hoi = hoi_seq_len, 4
        
        if mix_box_labels is not None:
            g_embeds, g_boxes, g_text_ids = GroundingInput.prepare_mixed_pipeline_input(tokenizer=self.tokenizer_2, text_encoder=self.text_encoder_2,
                                                                                        mix_boxes=mix_boxes, mix_box_labels=mix_box_labels, mix_hoi_labels=mix_hoi_labels,
                                                                                        hoi_seq_len=_hoi_seq_len, max_box=_max_box)
        else:
            g_embeds, g_boxes, g_text_ids = GroundingInput.prepare_pipeline_input(tokenizer=self.tokenizer_2, text_encoder=self.text_encoder_2,
                                                                                  boxes=boxes, box_labels=box_labels, hoi_labels=hoi_labels,
                                                                                  hoi_seq_len=_hoi_seq_len, max_box=_max_box, max_hoi=_max_hoi)
        g_embeds = g_embeds.to(dtype=self.dtype)
        g_boxes = g_boxes.to(dtype=self.dtype)
        g_text_ids = g_text_ids
        g_embeds = self.grounding_encoder(g_embeds, boxes=g_boxes, role_ids=g_text_ids[:, 2], idx_ids=g_text_ids[:, 1])

        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension
            
        # Prepare attention mask
        has_image_input = image_latents is not None
        img_token_width = int(width) // self.vae_scale_factor // 2
        img_token_height = int(height) // self.vae_scale_factor // 2
        # image is a image tensor of shape (B, C, H, W)
        cond_token_width = image.shape[3] // self.vae_scale_factor // 2 if has_image_input else 0
        cond_token_height = image.shape[2] // self.vae_scale_factor // 2 if has_image_input else 0
        
        ori_txt_tokens_size = prompt_embeds.shape[1]
        
        if arbitrary_masks is not None:
            processed_arbitrary_masks = GroundingInput.prepare_arbitrary_masks(arbitrary_masks, g_text_ids=g_text_ids,
                                                                                img_height=img_token_height,
                                                                                img_width=img_token_width,
                                                                                hoi_seq_len=_hoi_seq_len)
        
        attention_mask, neg_attention_mask = GroundingInput.prepare_attention_mask(
            out_text_ids=g_text_ids,
            out_boxes=g_boxes,
            img_tokens_size=latents.shape[1],
            txt_tokens_size=ori_txt_tokens_size,
            img_height=img_token_height,
            img_width=img_token_width,
            cond_tokens_size=image_latents.shape[1] if has_image_input else 0,
            cond_height=cond_token_height if has_image_input else 0,
            cond_width=cond_token_width if has_image_input else 0,
            arbitrary_masks=processed_arbitrary_masks if arbitrary_masks is not None else None,
            use_union_action_mask=use_union_action_mask,
        )
        # expand to attention heads
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.to(device=g_boxes.device)
        neg_attention_mask = neg_attention_mask.unsqueeze(1)
        neg_attention_mask = neg_attention_mask.to(device=g_boxes.device)
        
        # g_text_ids = torch.zeros_like(g_text_ids)  # we set HOI tokens RoPE ids to (0,0,0), align with text tokens
        # we set HOI tokens RoPE ids to (0, T+i, T+i) where T is max img token length, i is the index of the hoi slot
        # hoi_ids = g_text_ids[:, 1]
        g_text_ids = GroundingInput.get_rope_ids(g_text_ids,
                                                 img_height=img_token_height,
                                                 img_width=img_token_width,
                                                 cond_height=cond_token_height if has_image_input else 0,
                                                 cond_width=cond_token_width if has_image_input else 0)

        prompt_embeds = torch.cat([g_embeds, prompt_embeds], dim=1)
        text_ids = torch.cat([g_text_ids, text_ids], dim=0)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {
                "attention_mask": attention_mask,
            }
        elif "attention_mask" not in self.joint_attention_kwargs:
            self._joint_attention_kwargs["attention_mask"] = attention_mask

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self._joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs={"attention_mask": neg_attention_mask},
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):  # enforce xformers again, since we disable it previously, and VAE requires this
                image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


if __name__ == "__main__":
    
    grounding_encoder = GroundingEncoder()
    grounding_encoder.to("cuda", dtype=torch.bfloat16)
    
    pipeline = OneHOIPipeline.from_pretrained(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        grounding_encoder=grounding_encoder,
    )
    
    # Call the pipeline with example inputs
    output = pipeline(
        prompt=["A beautiful landscape",
                "A serene mountain view"],
        boxes=[[[100, 100, 200, 200], [300, 300, 400, 400]],
               [[100, 100, 200, 200], [300, 300, 400, 400], [100, 100, 200, 200], [300, 300, 400, 400]]],
        box_labels=[["tree", "mountain"],
                    ["tree", "mountain", "tree", "mountain"]],
        hoi_labels=[["look_at"],
                    ["look_at", "look_at"]],
    )
    output.images[0].save("test_pipeline_save.jpg")