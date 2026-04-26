import math
import random
from typing import Literal, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from diffusers.models.embeddings import get_fourier_embeds_from_boundingbox
from diffusers.utils import logging
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from transformers import T5TokenizerFast, T5EncoderModel


HOI_N_MAX = 4
BOX_N_MAX = 12 # 4x3
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
    
class GroundingInput:
    
    # Padding
    @staticmethod
    def pad_to(x, pad_shape, value=0):
        pad_size = list(pad_shape)
        pad_size[0] = pad_shape[0] - x.shape[0]
        if pad_size[0] > 0:
            pad = torch.full(pad_size, value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)
        return x
    
    @staticmethod
    @torch.no_grad()
    def _encode_prompt_with_t5(
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
        padding: Literal["max_length", "do_not_pad"] = "max_length",
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding=padding,
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        if hasattr(text_encoder, "module"):
            dtype = text_encoder.module.dtype
        else:
            dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds
    
    @classmethod
    def get_hoi_seq_len(cls, max_hoi_seq_len, total_hois):
        # Determine hoi sequence length
        # we maintain about total sequence = 6144 ? # this trigger OOM during training
        # or we should use 512 * 3 * 3 = 4608
        if total_hois <= 3:
            hoi_seq_len = 512
            max_hois = 3
        elif total_hois <= 6:
            hoi_seq_len = 256
            max_hois = 6
        elif total_hois <= 12:
            hoi_seq_len = 128
            max_hois = 12
        elif total_hois <= 24:
            hoi_seq_len = 64
            max_hois = 24
        elif total_hois <= 48:
            hoi_seq_len = 32
            max_hois = 48
        elif total_hois <= 96:
            hoi_seq_len = 16
            max_hois = 96
        elif total_hois <= 192:
            hoi_seq_len = 8
            max_hois = 192
        else:
            hoi_seq_len = 8
            max_hois = 192
            logger.warning(f"Number of HOIs ({total_hois}) exceeds the maximum limit of 192. Truncateing to 192.")

        hoi_seq_len = min(hoi_seq_len, max_hoi_seq_len)
        
        return hoi_seq_len, max_hois
    
    @classmethod
    def get_box_seq_len(cls, max_hoi_seq_len, total_boxes):
        hoi_seq_len, max_hois = cls.get_hoi_seq_len(max_hoi_seq_len, math.ceil(total_boxes / 3))
        return hoi_seq_len, max_hois * 3

    @classmethod
    def get_rope_ids(cls, g_text_ids, img_width: int = 64, img_height: int = 64, cond_width: int = 64, cond_height: int = 64):
        # Avoid in-place modification of the input tensor that may be needed for gradient computation
        max_img_dim = max(img_height, cond_height, img_width, cond_width)
        slot_ids = g_text_ids[:, 1]
        updated_cols = (slot_ids + max_img_dim).unsqueeze(1)
        g_text_ids = g_text_ids.clone()
        g_text_ids[:, 0] = 0  # set frame ids = 0
        g_text_ids[:, 1:] = updated_cols
        return g_text_ids

    @classmethod
    def get_prior(cls, sx: float, sy: float, h: int, w: int, device='cpu', dtype=torch.float32):
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        cx, cy = 0.5, 0.5
        eps = 1.0 / max(h, w)                    # avoid near-zero std
        sx_ = max(float(sx), eps)
        sy_ = max(float(sy), eps)

        prior = torch.exp(-(((X - cx) ** 2) / (2 * sx_ ** 2) +
                            ((Y - cy) ** 2) / (2 * sy_ ** 2)))
        prior /= prior.max()                      # normalize to max=1
        return prior


    @classmethod
    def preprocess_arbitrary_masks(cls, arbitrary_mask, img_height, img_width):
        # resize the arbitrary mask to img_height and img_width
        if arbitrary_mask is None:
            return None
        if not isinstance(arbitrary_mask, torch.Tensor):
            arbitrary_mask = torch.tensor(arbitrary_mask, dtype=torch.float32)
        # resize bool mask pytorch
        arbitrary_mask = arbitrary_mask.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)  # [1, 1, H, W]
        arbitrary_mask = F.interpolate(arbitrary_mask, size=(img_height, img_width), mode="bilinear", align_corners=False)
        arbitrary_mask = arbitrary_mask.squeeze(0).squeeze(0).to(dtype=torch.bool)
        return arbitrary_mask

    @classmethod
    def get_union_masks(cls, subject_mask, object_mask):
        if subject_mask is None or object_mask is None:
            return None
        if subject_mask.shape != object_mask.shape:
            raise ValueError(f"Shape mismatch: subject_mask {subject_mask.shape} and object_mask {object_mask.shape} must have the same shape.")
        
        subject_mask = subject_mask.to(torch.bool)
        object_mask  = object_mask.to(torch.bool)

        return subject_mask | object_mask
    
    @classmethod
    def prepare_arbitrary_masks(cls, arbitrary_masks: List[List[torch.Tensor]], g_text_ids: torch.Tensor,
                                img_height: int = 64, img_width: int = 64, hoi_seq_len: int = 64):
        """
        Prepare arbitrary masks for each box in the batch.
        
        Args:
            arbitrary_masks (List[List[torch.Tensor]]): List of batch samples, each containing a list of masks. \
                Each mask is a tensor of shape [img_tokens_size] or None. List must be in [B, N * M * T].
            boxes (torch.Tensor): Tensor of shape [B, N_max * M * T_max, 4] with box coordinates.
            img_height (int): Height of the image, default is 64.
            img_width (int): Width of the image, default is 64.
            hoi_seq_len (int): Maximum sequence length for HOI text encoding.

        Returns:
            List[List[torch.Tensor]]: Processed arbitrary masks with the same structure as input.
        """
        seq_len, _ = g_text_ids.shape
        batch_size = len(arbitrary_masks)
        for i in range(len(arbitrary_masks)):
            for j in range(len(arbitrary_masks[i])):
                for k in range(len(arbitrary_masks[i][j])):
                    if arbitrary_masks[i][j][k] is not None:
                        arbitrary_masks[i][j][k] = cls.preprocess_arbitrary_masks(arbitrary_masks[i][j][k], img_height, img_width)
            
        processed_masks = []
        for i in range(batch_size):
            sample_masks = []
            for j in range(seq_len):
                _, slot_id, role_id = g_text_ids[j]
                if role_id < 2:
                    mask = arbitrary_masks[i][slot_id][role_id] if arbitrary_masks[i][slot_id][role_id] is not None else None
                elif role_id == 2: # for action, it is the intersect of subject and object
                    mask = cls.get_union_masks(arbitrary_masks[i][slot_id][0], arbitrary_masks[i][slot_id][1])
                else:
                    raise ValueError(f"Invalid role_id {role_id} at batch {i}, index {j}")
                sample_masks.append(mask.flatten()) # flatten to [img_tokens_size]
            processed_masks.append(sample_masks)
        return processed_masks
    
    @classmethod
    def prepare_attention_mask(cls, out_text_ids: torch.Tensor, out_boxes: torch.Tensor,
                               img_tokens_size: int = 4096, txt_tokens_size: int = 512,
                               img_width: int = 64, img_height: int = 64,
                               cond_tokens_size: int = 4096, cond_width: int = 64, cond_height: int = 64,
                               arbitrary_masks: List[List[torch.Tensor]] = None, use_union_action_mask: bool = True):
        """
        Given input out_text_ids: [B, N_max * M * T_max, 3], and out_boxes: [B, N_max * M * T_max, 4]
        returns attention mask for the grounding encoder, where M could be 1 for object and 3 for HOI.
        
        Args:
            out_text_ids (torch.Tensor): Tensor of shape [B, N_max * M * T_max, 3] with text ids.
            out_boxes (torch.Tensor): Tensor of shape [B, N_max * M * T_max, 4] with box coordinates.
            img_tokens_size (int): Size of image tokens, default is 4096.
            txt_tokens_size (int): Size of text tokens, default is 512.
            img_width (int): Width of the image, default is 64.
            img_height (int): Height of the image, default is 64.
            cond_tokens_size (int): Size of condition tokens, default is 4096.
            cond_width (int): Width of the condition, default is 64.
            cond_height (int): Height of the condition, default is 64.
            arbitrary_masks (List[List[torch.Tensor]]): Optional list of arbitrary masks to apply. \
                Tensor is in shape [img_tokens_size] or None. List must be in [B, N * M * T].

        Returns:
            torch.Tensor: Attention mask of shape [B, N_max * M * T_max, N_max * M * T_max].
        """
        # assert shape of out_text_ids and out_boxes
        if out_text_ids.shape[0] != out_boxes.shape[1]:  
            raise ValueError(f"Shape mismatch: out_text_ids {out_text_ids.shape} and out_boxes {out_boxes.shape} must have the same sequence length.")
        assert img_tokens_size == img_width * img_height, \
            f"Image tokens size {img_tokens_size} must equal width {img_width} * height {img_height} = {img_width * img_height}"
        assert cond_tokens_size == cond_width * cond_height, \
            f"Condition tokens size {cond_tokens_size} must equal width {cond_width} * height {cond_height} = {cond_width * cond_height}"
        
        batch_size, seq_len, _ = out_boxes.shape

        mask_shape = seq_len + txt_tokens_size + img_tokens_size + cond_tokens_size
        all_img_tokens_size = img_tokens_size + cond_tokens_size
        attention_mask = torch.zeros(batch_size, mask_shape, mask_shape, dtype=torch.bool)

        # set image tokens attention mask to 1, last img_tokens_size tokens
        attention_mask[:, -all_img_tokens_size:, -all_img_tokens_size:] = 1

        # set text tokens attention mask to 1, first txt_tokens_size tokens
        attention_mask[:, seq_len:seq_len+txt_tokens_size, seq_len:seq_len+txt_tokens_size] = 1

        # set the cross attention mask for text tokens and image tokens
        attention_mask[:, seq_len:seq_len+txt_tokens_size, -all_img_tokens_size:] = 1
        attention_mask[:, -all_img_tokens_size:, seq_len:seq_len+txt_tokens_size] = 1
        
        # check if a token is valid, this can be obtained from box coordinates, it should be dropped if it is negative
        # we set attention mask of invalid one to False (at the end of this method)
        valid_seq = (out_boxes >= 0).all(dim=2).cpu()
        vq = valid_seq.unsqueeze(2)
        vk = valid_seq.unsqueeze(1)
        valid_seq = torch.ones([batch_size, seq_len, seq_len], dtype=torch.bool, device=attention_mask.device) & vq & vk

        # the text_ids could be in the form of:
        # tensor([[1, 0, 0],
        #          [1, 0, 1],
        #          [1, 0, 2],
        #          [1, 1, 0],
        #          [1, 1, 1],
        #          [1, 1, 2]]),
        # where the first dimension is the batch size, and the second dimension is the sequence length.
        # for the element with same second element, we set their attention mask to 1
        for i in range(batch_size):
            for j in range(seq_len):
                # Only compare with the seq_len tokens, and assign to the correct slice
                if out_text_ids[j, 0] == 0:  # if the first element is 0, it is a empty token and not valid
                    raise ValueError(f"Invalid token at batch {i}, index {j}: {out_text_ids[j]}")
                attention_mask[i, j, :seq_len] = (
                    out_text_ids[j, 1] == out_text_ids[:, 1]
                )
                
                # based on the out_boxes at the same index, set the attention mask, the boxes are in the form of:
                # [x1, y1, x2, y2]
                box = out_boxes[i, j]
                # verify box are valid and make sure both width and height not negative
                if (box >= 0).all() and (box[2] - box[0]) >= 0 and (box[3] - box[1]) >= 0:
                    if arbitrary_masks is not None and arbitrary_masks[i][j] is not None:
                        box_attn_mask = arbitrary_masks[i][j]
                        if box_attn_mask.numel() != img_tokens_size:
                            raise ValueError(f"Arbitrary mask at batch {i}, index {j} has incorrect size {box_attn_mask.numel()}, expected {img_tokens_size}")
                    elif box.sum() < 1e-6: # 2e-4 is min res for 64x64, 1e-6 is almost zero, here we want set zero box (randomly dropped box) with all attended
                        box_attn_mask = torch.ones(img_tokens_size, dtype=torch.bool)
                    elif use_union_action_mask and out_text_ids[j, 2] == 2: # for action, we use the union of subject and object
                        # we assume both direction of attention is same, thus we take from one only
                        subject_index = (out_text_ids[:, 2] == 0) & (out_text_ids[:, 1] == out_text_ids[j, 1])
                        subject_index = subject_index.to(device=attention_mask.device)
                        object_index = (out_text_ids[:, 2] == 1) & (out_text_ids[:, 1] == out_text_ids[j, 1])
                        object_index = object_index.to(device=attention_mask.device)
                        if img_tokens_size == all_img_tokens_size:
                            subject_attn_mask = attention_mask[i, -img_tokens_size:, :seq_len][:, subject_index]
                            object_attn_mask  = attention_mask[i, -img_tokens_size:, :seq_len][:, object_index]
                        else:
                            subject_attn_mask = attention_mask[i, -all_img_tokens_size:-all_img_tokens_size+img_tokens_size, :seq_len][:, subject_index]
                            object_attn_mask  = attention_mask[i, -all_img_tokens_size:-all_img_tokens_size+img_tokens_size, :seq_len][:, object_index]
                        if subject_attn_mask.numel() == 0 or object_attn_mask.numel() == 0:
                            box_attn_mask = torch.ones(img_tokens_size, dtype=torch.bool)
                        else:
                            subject_attn_mask = subject_attn_mask.any(dim=1)
                            object_attn_mask  = object_attn_mask.any(dim=1)
                            box_attn_mask = subject_attn_mask | object_attn_mask
                    else:
                        box_attn_mask = torch.zeros(img_tokens_size, dtype=torch.bool)
                        box_attn_mask = box_attn_mask.reshape(img_height, img_width)
                        x1_idx = int(box[0] * img_width)
                        y1_idx = int(box[1] * img_height)
                        x2_idx = int(box[2] * img_width)
                        y2_idx = int(box[3] * img_height)
                        # Make the end indices inclusive, but clamp to image size
                        x2_idx = min(x2_idx, img_width - 1)
                        y2_idx = min(y2_idx, img_height - 1)
                        # Add 1 to end indices for inclusive slicing
                        box_attn_mask[y1_idx:y2_idx+1, x1_idx:x2_idx+1] = 1
                    # flatten the box attention mask to match the img_tokens_size
                    box_attn_mask = box_attn_mask.flatten()
                    
                    # set the attention mask for the box tokens
                    if img_tokens_size == all_img_tokens_size:
                        attention_mask[i, -img_tokens_size:, j] = box_attn_mask
                        attention_mask[i, j, -img_tokens_size:] = box_attn_mask
                    else:
                        attention_mask[i, j, -all_img_tokens_size:-all_img_tokens_size+img_tokens_size] = box_attn_mask
                        attention_mask[i, -all_img_tokens_size:-all_img_tokens_size+img_tokens_size, j] = box_attn_mask
            
            # For HOI, prevent S to attend to O and vice versa
            roles   = out_text_ids[:, 2]
            is_S = roles == 0
            is_O = roles == 1
            # is_A = roles == 2
            forbid_SO = (is_S[:, None] & is_O[None, :]) | (is_O[:, None] & is_S[None, :])
            forbid = forbid_SO
            forbid = forbid.to(device=attention_mask.device)
            attention_mask[:, :seq_len, :seq_len] &= ~forbid
            
        # set invalid one to False
        # this seems to cause NaN, because for some invalid query, all its key now become 0        
        # we can fix it with minimal self-attention via a diagonal mask
        eye = torch.eye(seq_len, device=attention_mask.device, dtype=torch.bool)[None]
        attention_mask[:, :seq_len, :seq_len] = attention_mask[:, :seq_len, :seq_len].bool() & valid_seq[:, :seq_len, :seq_len] | eye
        negative_mask = attention_mask[:, seq_len:, seq_len:]
        return attention_mask, negative_mask

    @classmethod
    def prepare_train_input(cls,
                            tokenizer, text_encoder,
                            boxes=None, hois=None, objects=None,
                            random_drop_boxes: float = 0.0, random_drop_hois: float = 0.0, hoi_seq_len: int = 64):
        """
        Prepares input for the grounding encoder during training.
        Args:
            tokenizer (T5TokenizerFast): Tokenizer for encoding text.
            boxes (List[List[List[float]]]): List of batch samples, each containing a list of boxes, each box as [x1, y1, x2, y2].
            hois (List[List[dict]]): List of batch samples, each containing a list of HOI labels.
            objects (List[List[dict]]): List of batch samples, each containing a list of object labels.
            random_drop_boxes (float): Probability of randomly dropping boxes during training.
            random_drop_hois (float): Probability of randomly dropping HOIs during training.
            hoi_seq_len (int): Maximum sequence length for HOI text encoding.
            max_box (int): Maximum number of boxes to consider.
            max_hoi (int): Maximum number of HOIs to consider.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - out_embeds: Tensor of shape [B, T_max, D] with text embeddings.
                - out_boxes: Tensor of shape [B, T_max, 4] with box coordinates.
                - out_text_ids: Tensor of shape [B, T_max, 3] with text ids.
        """
        # If no boxes, hois, or objects are provided, return None
        if boxes[0] is None and hois[0] is None and objects[0] is None:
            return None, None, None
        
        if random.random() < random_drop_boxes:
            # replace boxes with zeros, maintaining the shape,
            # the shape is [B, N, 1, 4] or [B, N, 2, 4] for HOI
            # for each B x tensor(N,1,4) or B x tensor(N,2,4)
            # print(boxes)
            # print(boxes[0].shape)
            for sample in boxes:
                if isinstance(sample, torch.Tensor):
                    sample.fill_(0.0)
                else:
                    raise ValueError("boxes should be a list of tensors or None, got: {}. HOI: {} BOX:{}".format(
                        type(sample),
                        objects[0] is None,
                        boxes[0] is not None))
            # print(f"Randomly dropping boxes, replaced with zeros. hoi:{objects[0] is None} box:{objects[0] is not None}")
            # print(boxes)

        # HOI case: hois provided, objects are None
        if boxes[0] is not None and hois[0] is not None and objects[0] is None:
            
            if random.random() < random_drop_hois:
                box_labels = cls.obtain_only_box_labels_from_hoi(hois)
                # reshape boxes from hoi to independent boxes, to reduce it into Box Generation task.
                # Flatten boxes from [B, N, M, 4] to [B, N*M, 4] for each sample, then unsqueeze to [B, N*M, 1, 4]
                boxes = [
                    sample.reshape(-1, 4) if isinstance(sample, torch.Tensor) else torch.tensor(sample, dtype=torch.float32).reshape(-1, 4)
                    for sample in boxes
                ]
                # boxes = [[torch.tensor(box, dtype=torch.float32) for box in sample] for sample in boxes]  # keep as list of lists for variable N
                boxes = [sample.unsqueeze(1) for sample in boxes]  # [B][N*M, 1, 4]
                _hoi_seq_len, _max_box = cls.get_box_seq_len(hoi_seq_len, total_boxes=max(len(h) for h in box_labels))
                processed_boxes, box_prompt_embeds = cls.preprocess_box(boxes, box_labels, tokenizer, text_encoder, _hoi_seq_len, max_box=_max_box)
                return cls.prepare_box(box_prompt_embeds, processed_boxes, _max_box)
            else:
                box_labels, hoi_labels = cls.obtain_box_hoi_labels(hois)
                _hoi_seq_len, _max_hoi = cls.get_hoi_seq_len(hoi_seq_len, max(len(h) for h in hoi_labels) if hoi_labels is not None else 0)
                processed_boxes, box_prompt_embeds = cls.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder, _hoi_seq_len, max_hoi=_max_hoi)
                return cls.prepare_hoi(box_prompt_embeds, processed_boxes, _max_hoi)

        # If boxes and objects are provided
        elif boxes[0] is not None and objects[0] is not None and hois[0] is None:
            box_labels = cls.obtain_box_labels(objects)
            _hoi_seq_len, _max_box = cls.get_box_seq_len(hoi_seq_len, total_boxes=max(len(h) for h in box_labels))
            processed_boxes, box_prompt_embeds = cls.preprocess_box(boxes, box_labels, tokenizer, text_encoder, _hoi_seq_len, max_box=_max_box)
            return cls.prepare_box(box_prompt_embeds, processed_boxes, _max_box)

        else:
            raise ValueError(f"Unexpected case of boxes={'None' if boxes is None else 'Not None'},"
                             f" objects={'None' if objects is None else 'Not None'},"
                             f" hois={'None' if hois is None else 'Not None'}.")


    @classmethod
    def prepare_mixed_pipeline_input(cls, tokenizer, text_encoder,
                                     mix_boxes=None, mix_box_labels=None, mix_hoi_labels=None,
                                     hoi_seq_len: int = 64, max_box: int | None = None):
        """
        Prepares input for the grounding encoder pipeline, accepting arbitary modality.
        Args:
            tokenizer (T5TokenizerFast): Tokenizer for encoding text.
            text_encoder (T5EncoderModel): Text encoder model for encoding tokenized text.
            mix_boxes (List[List[List[float]]]): List of batch samples with mixed boxes or None.
            mix_box_labels (List[List[str]]): List of batch samples with mixed box labels or None.
            mix_hoi_labels (List[List[str]]): List of batch samples with mixed HOI labels or None.
            hoi_seq_len (int): Maximum sequence length for HOI text encoding.
            max_box (int): Maximum number of boxes to consider.
            max_hoi (int): Maximum number of HOIs to consider.
        """
        if mix_boxes is None and mix_box_labels is None and mix_hoi_labels is None:
            return None, None, None
        
        processed_boxes, box_prompt_embeds = cls.preprocess_mixed(mix_boxes, mix_box_labels, mix_hoi_labels, tokenizer, text_encoder, hoi_seq_len, max_box=max_box)
        out_embeds, out_boxes, out_text_ids = cls.prepare_mixed(box_prompt_embeds, processed_boxes, max_box=max_box)
        
        return out_embeds, out_boxes, out_text_ids
    
    @classmethod
    def preprocess_mixed(cls, mix_boxes, mix_box_labels, mix_hoi_labels, tokenizer, text_encoder, hoi_seq_len, max_box=None):
        """
        Preprocesses mixed boxes and labels for the grounding encoder.
        Args:
            mix_boxes (List[List[List[float]]]): List of batch samples with mixed boxes or None. The shape must be [B, N, M] x [4 or None]
            mix_box_labels (List[List[str]]): List of batch samples with mixed box labels or None. The shape must be [B, N, M] x (str or None)
            mix_hoi_labels (List[List[str]]): List of batch samples with mixed HOI labels or None. The shape must be [B, N, 1] x (str or None). If n-th box_labels has M=3, then this must be str.
            tokenizer (T5TokenizerFast): Tokenizer for encoding text.
            text_encoder (T5EncoderModel): Text encoder model for encoding tokenized text.
            hoi_seq_len (int): Maximum sequence length for HOI text encoding.
            max_box (int): Maximum number of boxes to consider.
        """
        assert len(mix_boxes) == len(mix_box_labels) == len(mix_hoi_labels), \
            f"Batch size mismatch: mix_boxes {len(mix_boxes)}, mix_box_labels {len(mix_box_labels)}, mix_hoi_labels {len(mix_hoi_labels)}"
        for b in range(len(mix_boxes)):
            assert len(mix_boxes[b]) == len(mix_box_labels[b]) == len(mix_hoi_labels[b]), \
                f"HOI instance number mismatch at index {b}: mix_boxes {len(mix_boxes[b])}, mix_box_labels {len(mix_box_labels[b])}, mix_hoi_labels {len(mix_hoi_labels[b])}"
            for n in range(len(mix_boxes[b])):
                assert len(mix_boxes[b][n]) == len(mix_box_labels[b][n]), \
                    f"Role (subject/object/action) mismatch at index {b},{n}: mix_boxes {len(mix_boxes[b][n])}, mix_box_labels {len(mix_box_labels[b][n])}"
                if len(mix_boxes[b][n]) == 2:
                    assert isinstance(mix_hoi_labels[b][n], str), \
                        f"HOI label must be str when box_labels has 2 roles at index {b},{n}: got {type(mix_hoi_labels[b][n])}"
                else:
                    assert mix_hoi_labels[b][n] is None, \
                        f"HOI label must be None when box_labels has not 2 roles at index {b},{n}: got {mix_hoi_labels[b][n]}"
                assert len(mix_boxes[b][n]) in [1, 2], \
                    f"Number of roles (subject/object/action) must be 1 or 2 for Object or HOI instance at index {b},{n}: got {len(mix_boxes[b][n])}"
        
        B = len(mix_boxes)
        processed_boxes = []
        box_prompt_embeds = []
        for b in range(B):
            N = len(mix_boxes[b])
            box_list = []
            box_prompt_list = []
            for n in range(N):
                M = len(mix_boxes[b][n])  # M is 1 for object and 2 for HOI
                for m, box in enumerate(mix_boxes[b][n]):
                    if box is None:
                        mix_boxes[b][n][m] = [0.0, 0.0, 0.0, 0.0]
                if M == 2:
                    # get_action_boxes input is [B, N, 2, 4]
                    _boxes = torch.tensor(mix_boxes[b][n], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 2, 4]
                    action_box = cls.get_enclosing_action_boxes(_boxes)[0, 0].tolist()  # [4]
                    mix_boxes[b][n].append(action_box)
                    mix_box_labels[b][n].append(mix_hoi_labels[b][n])
                    
                embs = cls._encode_prompt_with_t5(
                    text_encoder, tokenizer, device=text_encoder.device,
                    prompt=mix_box_labels[b][n],
                    padding="max_length", max_sequence_length=hoi_seq_len
                )  # [chunk, T, D]

                box_list.append(mix_boxes[b][n])  # proc_box is a list of length B=1
                box_prompt_list.append(embs)  # box_prompt is a list of length B=1
            # Concatenate all boxes and prompts for this batch item
            processed_boxes.append(box_list)  # box_list: [N*M, 4]
            box_prompt_embeds.append(box_prompt_list) # box_prompt_list: [N*M, T, D]
        # processed_boxes: [B, N*M, 4]
        # box_prompt_embeds: [B, N*M, T, D]
        return processed_boxes, box_prompt_embeds
            
    @classmethod
    def prepare_mixed(cls, box_prompt_embeds, processed_boxes, max_box=None):
        """
        Prepares mixed box embeddings and coordinates for the grounding encoder.
        Args:
            box_prompt_embeds (List[List[torch.Tensor]]): List of batch samples, each containing a list of box prompt embeddings.
            processed_boxes (List[List[List[float]]]): List of batch samples, each containing a list of boxes, each box as [x1, y1, x2, y2].
            max_box (int): Maximum number of boxes to consider.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - out_embeds: Tensor of shape [B, T_max, D] with text embeddings.
                - out_boxes: Tensor of shape [B, T_max, 4] with box coordinates.
                - out_text_ids: Tensor of shape [B, T_max, 3] with text ids.
        """
        B = len(processed_boxes)
        # if B is > 1,
        # when N is variable, then we must set all to N_max, and pad the missing ones
        # when M is variable (1 or 3), we must set all to 3, and pad the missing ones
        # assert B == 1, f"Batch size must be 1 for mixed input, got {B=}"

        T = box_prompt_embeds[0][0][0].shape[0]  # Assuming all have the same T
        D = box_prompt_embeds[0][0][0].shape[-1]  # Assuming all have the same D
        device = box_prompt_embeds[0][0][0].device
            
        if B > 1:
            N_max = min(max_box, max(len(b) for b in processed_boxes))
            
            out_embeds = torch.zeros((B, N_max * 3 * T, D), dtype=torch.float32, device=device)
            out_boxes = torch.full((B, N_max * 3 * T, 4), -1.0, dtype=torch.float32, device=device)
            
            _M = 3
            ones = torch.full((N_max, _M, T, 1), 8, dtype=torch.long, device=device) # it was 1, now 8
            ns = torch.arange(N_max, device=device).view(N_max, 1, 1, 1).expand(N_max, _M, T, 1)
            ms = torch.arange(_M, device=device).view(1, _M, 1, 1).expand(N_max, _M, T, 1)
            ids = torch.cat([ones, ns, ms], dim=-1)  # [N_max, M, T_max, 3]
            
            for i in range(B):
                N = min(len(processed_boxes[i]), N_max)
                for n in range(N):
                    M = len(processed_boxes[i][n])  # M is 1 for object and 3 for HOI
                    for m in range(M):
                        out_embeds[i, (n * _M + m) * T : (n * _M + m + 1) * T] = box_prompt_embeds[i][n][m]
                        out_boxes[i, (n * _M + m) * T : (n * _M + m + 1) * T] = torch.tensor(processed_boxes[i][n][m],
                                                                                             dtype=torch.float32, device=device).unsqueeze(0).expand(T, 4)
            # out_embeds: [B, N_max*M*T, D]
            # out_boxes: [B, N_max*M*T, 4]
            # out_text_ids: [N_max*M*T, 3]
            out_embeds = out_embeds
            out_boxes = out_boxes
            out_text_ids = ids.view(N_max * _M * T, 3)
        elif B == 1:
            # this we could handle variable N and M
            N = len(processed_boxes[0])
            NM = sum(len(b) for b in processed_boxes[0])  # N * M
            out_embeds = torch.zeros((1, NM * T, D), dtype=torch.float32, device=device)
            out_boxes = torch.full((1, NM * T, 4), -1.0, dtype=torch.float32, device=device)
            out_text_ids = torch.zeros((NM * T, 3), dtype=torch.long, device=device)
            out_text_ids[:, 0] = 8  # it was 1, now 8
            
            nm = 0
            for n in range(N):
                M = len(processed_boxes[0][n])  # M is 1 for object and 3 for HOI
                for m in range(M):
                    out_embeds[0, nm * T : (nm + 1) * T] = box_prompt_embeds[0][n][m]
                    out_boxes[0, nm * T : (nm + 1) * T] = torch.tensor(processed_boxes[0][n][m],
                                                                      dtype=torch.float32, device=device).unsqueeze(0).expand(T, 4)
                    out_text_ids[nm * T : (nm + 1) * T, 1] = n
                    out_text_ids[nm * T : (nm + 1) * T, 2] = m
                    nm += 1
            # out_embeds: [B, N*M*T, D]
            # out_boxes: [B, N*M*T, 4]
            # out_text_ids: [N*M*T, 3]
        else:
            raise ValueError(f"Batch size must not be 0, got {B=}")
        return out_embeds, out_boxes, out_text_ids

    @classmethod
    def prepare_pipeline_input(cls, tokenizer, text_encoder,
                               boxes=None, box_labels=None, hoi_labels=None,
                               hoi_seq_len: int = 64,
                               max_hoi: int | None = None, max_box: int | None = None):
        """
        Prepares input for the grounding encoder pipeline.
        Deterministic, no random drop of boxes and hois for inference.
        Args:
            tokenizer (T5TokenizerFast): Tokenizer for encoding text.
            text_encoder (T5EncoderModel): Text encoder model for encoding tokenized text.
            boxes (List[List[List[float]]]): List of batch samples, each containing a list of boxes, each box as [x1, y1, x2, y2].
            box_labels (List[List[str]]): List of batch samples, each containing a list of labels for boxes.
            hoi_labels (List[List[str]]): List of batch samples, each containing a list of HOI labels.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - out_embeds: Tensor of shape [B, T_max, D] with text embeddings.
                - out_boxes: Tensor of shape [B, T_max, 4] with box coordinates.
                - out_text_ids: Tensor of shape [B, T_max, 3] with text ids.
        Raises:
            ValueError: If boxes, box_labels, and hoi_labels combo are unexpected.
        """
        if boxes is None and box_labels is None and hoi_labels is None:
            return None, None, None
        
        # boxes is [[sub1, obj2], ...]
        if hoi_labels is not None and box_labels is not None and boxes is not None:
            max_hoi = max_hoi if max_hoi is not None else HOI_N_MAX
            processed_boxes, box_prompt_embeds = cls.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder, hoi_seq_len, limit_max_hoi=False, max_hoi=max_hoi)
            return cls.prepare_hoi(box_prompt_embeds, processed_boxes, max_hoi=max_hoi)
        elif hoi_labels is not None and box_labels is not None and boxes is None:
            # Create dummy subject and object boxes of zeros for each HOI in each batch
            max_hoi = max_hoi if max_hoi is not None else HOI_N_MAX
            boxes = [
                [
                    [0.0, 0.0, 0.0, 0.0] for _ in range(len(hois) * 2)
                ]
                for hois in hoi_labels
            ]
            processed_boxes, box_prompt_embeds = cls.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder, hoi_seq_len, limit_max_hoi=False, max_hoi=max_hoi)
            return cls.prepare_hoi(box_prompt_embeds, processed_boxes, max_hoi=max_hoi)
        elif boxes is not None and box_labels is not None and hoi_labels is None:
            max_box = max_box if max_box is not None else BOX_N_MAX
            processed_boxes, box_prompt_embeds = cls.preprocess_box(boxes, box_labels, tokenizer, text_encoder, hoi_seq_len, limit_max_box=False, max_box=max_box)
            return cls.prepare_box(box_prompt_embeds, processed_boxes, max_box=max_box)
        elif boxes is None and box_labels is not None and hoi_labels is None:
            max_box = max_box if max_box is not None else BOX_N_MAX
            boxes = [
                [
                    [0.0, 0.0, 0.0, 0.0] for _ in range(len(labels))
                ]
                for labels in box_labels
            ]
            processed_boxes, box_prompt_embeds = cls.preprocess_box(boxes, box_labels, tokenizer, text_encoder, hoi_seq_len, limit_max_box=False, max_box=max_box)
            return cls.prepare_box(box_prompt_embeds, processed_boxes, max_box=max_box)
        else:
            raise ValueError(f"Unexpected case of boxes={'None' if boxes is None else 'Not None'},"
                             f" box_labels={'None' if box_labels is None else 'Not None'},"
                             f" hoi_labels={'None' if hoi_labels is None else 'Not None'}.")

    @classmethod
    def obtain_only_box_labels_from_hoi(cls, batch_hois: List[List[Dict]]):
        """
        Extracts only box labels from a batch of HOIs.
        This is used when we randomly drop HOI labels.
        
        Args:
            batch_hois (List[List[Dict]]): List of batch samples, each containing a list of HOI dictionaries.
        Returns:
            List[List[str]]: List of box labels for each batch sample.
        """
        box_labels = []
        for batch in batch_hois:
            batch_box_labels = []
            for hoi in batch:
                batch_box_labels.extend([hoi['subject']])
                batch_box_labels.extend([hoi['object']])
            box_labels.append(batch_box_labels)
        return box_labels
    
    @classmethod
    def obtain_box_hoi_labels(cls, batch_hois: List[List[Dict]]):
        box_labels = []
        hoi_labels = []
        for batch in batch_hois:
            batch_box_labels = []
            batch_hoi_labels = []
            for hoi in batch:
                batch_box_labels.extend([hoi['subject'], hoi['object']])
                batch_hoi_labels.extend([hoi['action']])
            box_labels.append(batch_box_labels)
            hoi_labels.append(batch_hoi_labels)
        return box_labels, hoi_labels
    
    @classmethod
    def obtain_box_labels(cls, batch_objs: List[List[Dict]]):
        box_labels = []
        for batch in batch_objs:
            batch_box_labels = []
            for obj in batch:
                batch_box_labels.extend([obj['phrases']])
            box_labels.append(batch_box_labels)
        return box_labels

    @classmethod
    def preprocess_box(cls, boxes, box_labels, tokenizer, text_encoder, hoi_seq_len=64, limit_max_box=True, max_box=BOX_N_MAX):
        """
        Prepares box data for the grounding encoder.
        Args:
            boxes (List[List[List[float]]]): List of batch samples, each containing a list of boxes, each box as [x1, y1, x2, y2].
            box_labels (List[List[str]]): List of batch samples, each containing a list of labels for boxes.
            tokenizer: Tokenizer object used to tokenize box labels.
            text_encoder: Text encoder model used to encode tokenized labels.
        Returns:
            processed_boxes (List[List[List[List[float]]]]): List of batch samples, each containing a list of boxes, each box wrapped in a list as [[x1, y1, x2, y2]].
            box_prompt_embeds (List[List[List[Tensor]]]): List of batch samples, each containing a list of boxes, each box as a list containing a tensor of shape [T', D] for token embeddings.
        """
        B = len(boxes)
        processed_boxes = []
        box_prompt_embeds = []
        N = max_box if limit_max_box else max(len(b) for b in boxes)  # number of boxes per sample, max 12
        for b in range(B):
            token_budget = 9 * 512
            chunk_size = max(1, token_budget // hoi_seq_len)
            all_embs = []
            box_texts = box_labels[b][:N]
            _local_N = len(box_texts)
            for start in range(0, len(box_texts), chunk_size):
                chunk = box_texts[start:start + chunk_size]
                chunk_embs = cls._encode_prompt_with_t5(
                    text_encoder, tokenizer, device=text_encoder.device,
                    prompt=chunk, padding="max_length", max_sequence_length=hoi_seq_len
                )  # [chunk, T, D]
                all_embs.append(chunk_embs)
            embs = torch.cat(all_embs, dim=0)  # [N*M, T, D]
            embs_ = embs.reshape(_local_N, 1, hoi_seq_len, -1)
                
            box_prompt_embeds.append(embs_) # this has to be [B, N, 1, T, D]
            if isinstance(boxes[b], torch.Tensor) and boxes[b].ndim == 3: # already in [N, 1, 4]
                processed_boxes.append(boxes[b][:N]) # act like no-op
            else:
                processed_boxes.append([[box] for box in boxes[b][:N]])
        # processed_boxes should be [B, N ,1, 4]
        return processed_boxes, box_prompt_embeds

    @classmethod
    def prepare_box(cls, box_prompt_embeds, boxes, max_box=BOX_N_MAX):
        """
        Prepares box features for the grounding encoder.
        boxes: [B, N, 1, 4] where last dim is [x1, y1, x2, y2]
        box_prompt_embeds: Tensor where each is a sample in Tensor [B, N, M, T, D] representing token embeddings for each box.
        Outputs:
        - out_embeds: [B, N_max * 1 * T, D]
        - out_boxes: [B, N_max * 1 * T_max, 4]
        - out_text_ids: [B, N_max * 1 * T_max, 3]
        """
        B = len(boxes)
        N_max, M = max_box, 1  # max boxes per sample, 1 box per interaction
        N = min(N_max, max(len(b) for b in boxes))  # number of boxes per sample
        T = box_prompt_embeds[0][0][0].shape[0] # 64 or cfg.model.hoi_max_seq_len
        D = box_prompt_embeds[0][0][0].shape[-1] # 4096
        device = box_prompt_embeds[0][0][0].device

        # Allocate outputs
        out_embeds = torch.zeros((B, N, M, T, D), device=device)
        out_boxes  = torch.full((B, N, M, T, 4), -1.0, device=device)
        # Text ids for boxes, [B, N, M, T, 3]
        ones = torch.full((N, M, T, 1), 8, dtype=torch.long, device=device) # it was 1, now 8
        ns = torch.arange(N, device=device).view(N, 1, 1, 1).expand(N, M, T, 1)
        ms = torch.arange(M, device=device).view(1, M, 1, 1).expand(N, M, T, 1)
        ids = torch.cat([ones, ns, ms], dim=-1)  # [N, M, T, 3]
        ids = ids.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, N, M, T, 3]

        for i in range(B):
            N_i = min(len(box_prompt_embeds[i]), N)
            if isinstance(boxes[i], torch.Tensor):
                boxes_tensor = boxes[i][:N_i] # N, 4
            else:
                boxes_tensor = torch.tensor(boxes[i][:N_i], dtype=torch.float32, device=device)  # [N, 4]
            
            for n in range(N_i):
                for m in range(M):
                    emb = box_prompt_embeds[i][n][m]  # [T', D]
                    out_embeds[i, n, m] = emb
                    # boxes[i][n, m, :] is [4], expand to [T, 4]
                    out_boxes[i, n, m] = boxes_tensor[n, m].expand(T, 4)
            # Padding happens automatically

        # Reshape to [B, N_max * M * T_max, ...]
        out_embeds = out_embeds.view(B, N * M * T, D)
        out_boxes = out_boxes.view(B, N * M * T, 4)
        out_text_ids = ids.view(B, N * M * T, 3)[0]
        return out_embeds, out_boxes, out_text_ids
    
    @classmethod
    def preprocess_hoi(cls, boxes, box_labels, hoi_labels, tokenizer, text_encoder, hoi_seq_len=64, limit_max_hoi=True, max_hoi=HOI_N_MAX):
        """
        Prepares HOI data for the grounding encoder.

        Args:
            boxes (List[List[float]]): List of lists of lists of floats, where each innermost list contains [x1, y1, x2, y2] for subject and object boxes.
            box_labels (List[List[str]]): List of lists of strings, each inner list contains labels for boxes per interaction.
            hoi_labels (List[List[str]]): List of lists of strings, each inner list contains HOI labels per interaction.
            tokenizer: Tokenizer object used to tokenize box and HOI labels.
            text_encoder: Text encoder model used to encode tokenized labels.

        Returns:
            boxes (List[List[List[float]]]): Tensor of shape [B, N, 3, 4] where last dim is [x1, y1, x2, y2].
            box_prompt_embeds (List[List[List[Tensor]]]): List of lists of lists of Tensors, where each Tensor is [3, D] for each box.
        """
        
        # boxes: [B, N*2, 4] -> [B, N, 2, 4]
        # box_labels: List[List[str]], hoi_labels: List[List[str]]
        # For each interaction, create a dict with subject, object, action
        B = len(boxes)
        M = 3
        if isinstance(boxes[0], torch.Tensor) and boxes[0].ndim == 3: # if already in [N, 2, 4], reshape back to [N*2, 4]
            boxes = [b.reshape(-1, 4) for b in boxes]
        
        processed_boxes = []
        box_prompt_embeds = []
        for b in range(B):
            N = min(len(boxes[b]) // 2, max_hoi) if limit_max_hoi else len(boxes[b]) // 2  # number of interactions, max 4
            boxes_b = []
            hoi_texts = []
            for n in range(N):
                hoi_texts.extend([box_labels[b][n*2], box_labels[b][n*2+1], hoi_labels[b][n]]) # subject, object, action
                subject_box = boxes[b][n*2]
                object_box = boxes[b][n*2+1]
                boxes_b.append([subject_box, object_box])
            # hoi_texts may be large; batch them to save GPU memory.
            # At hoi_seq_len=512 we can handle 4 HOIs -> token budget = 4 * 512 = 2048 "token-units".
            token_budget = 9 * 512
            chunk_size = max(1, token_budget // hoi_seq_len)

            all_embs = []
            for start in range(0, len(hoi_texts), chunk_size):
                chunk = hoi_texts[start:start + chunk_size]
                chunk_embs = cls._encode_prompt_with_t5(
                    text_encoder, tokenizer, device=text_encoder.device,
                    prompt=chunk, padding="max_length", max_sequence_length=hoi_seq_len
                )  # [chunk, T, D]
                all_embs.append(chunk_embs)

            embs = torch.cat(all_embs, dim=0)  # [N*M, T, D]
            embs_ = embs.reshape(N, M, hoi_seq_len, -1)
            processed_boxes.append(boxes_b)
            box_prompt_embeds.append(embs_)
        return processed_boxes, box_prompt_embeds
    
    @classmethod
    def prepare_hoi(cls, box_prompt_embeds, boxes, max_hoi=HOI_N_MAX):
        """
        Enforces:
        - max 8 interactions per sample
        - 3 boxes per interaction (subject, object, action)
        - 10 tokens per box
        Outputs all tensors with shape [B, 10*3*10, ...]
        Prepares box features for the grounding encoder.
        boxes: [B, N, 2, 4] where last dim is [x1, y1, x2, y2] and dim=2 indexes subject and object.
        box_prompt_embeds: List[List[List[Tensor]]] where each Tensor is [T', D] for each box.
        Outputs:
        - out_embeds: [B, N_max * M * T_max, D]
        - out_boxes: [B, N_max * M * T_max, 4]
        - out_text_ids: [B, N_max * M * T_max, 3
        """
        B = len(boxes)
        N_max, M = max_hoi, 3  # interactions, boxes
        N_max = min(N_max, max(len(b) for b in boxes))  # number of boxes per sample
        T = box_prompt_embeds[0][0][0].shape[0]  # T_max, e.g. 64
        D = box_prompt_embeds[0][0][0].shape[-1]
        device = box_prompt_embeds[0][0][0].device

        # Allocate outputs
        out_embeds = torch.zeros((B, N_max, M, T, D), device=device)
        out_boxes  = torch.full((B, N_max, M, T, 4), -1.0, device=device)
        # out_text_ids = torch.full((B, N_max, M, T_max, 3), -1, dtype=torch.long, device=boxes[0].device)
        ones = torch.full((N_max, M, T, 1), 8, dtype=torch.long, device=device) # it was 1, now 8
        ns = torch.arange(N_max, device=device).view(N_max, 1, 1, 1).expand(N_max, M, T, 1)
        ms = torch.arange(M, device=device).view(1, M, 1, 1).expand(N_max, M, T, 1)
        ids = torch.cat([ones, ns, ms], dim=-1)  # [N_max, M, T_max, 3]
        ids = ids.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, N_max, M, T_max, 3]

        for i in range(B):
            N = min(len(box_prompt_embeds[i]), N_max)
            if isinstance(boxes[i], torch.Tensor):
                so_boxes = boxes[i][:N]
            elif isinstance(boxes[i], list) and isinstance(boxes[i][0], list): # List[List[Tensor]], B, N, 2 is nested list of [4] tensor
                # Convert to tensor [N, 2, 4], assuming boxes[i] is a list of lists of tensors [4]
                so_boxes = torch.stack(
                    [torch.stack([torch.as_tensor(b, dtype=torch.float32) for b in pair], dim=0)
                    for pair in boxes[i]],
                    dim=0
                )
            else:
                so_boxes = torch.tensor(boxes[i][:N], dtype=torch.float32, device=device)  # [N, 2, 4]
            action_boxes = cls.get_enclosing_action_boxes(so_boxes.unsqueeze(0)).squeeze(0)  # [N, 4] # Union
            
            soa_boxes = torch.cat([so_boxes, action_boxes.unsqueeze(1)], dim=1)  # [N, 3, 4]
            for n in range(N):
                for m in range(M):
                    emb = box_prompt_embeds[i][n][m]  # [T', D]
                    # Fill the valid parts
                    out_embeds[i, n, m] = emb
                    out_boxes[i, n, m] = soa_boxes[n, m].expand(T, 4)

        # Reshape to [B, 300, ...]
        out_embeds = out_embeds.view(B, N_max * M * T, D)
        out_boxes = out_boxes.view(B, N_max * M * T, 4)
        out_text_ids = ids.view(B, N_max * M * T, 3)[0]
        return out_embeds, out_boxes, out_text_ids

    @classmethod
    def get_action_boxes(cls, boxes):
        """
        Compute action boxes using the 'between' operation.

        Args:
            boxes (torch.Tensor): Tensor of shape [B, N, 2, 4], where the last dimension 
                                is [x1, y1, x2, y2] and dim=2 indexes subject and object.

        Returns:
            torch.Tensor: Action boxes of shape [B, N, 4]
        """
        subj_boxes = boxes[:, :, 0, :]  # [B, N, 4]
        obj_boxes = boxes[:, :, 1, :]   # [B, N, 4]

        all_x = torch.cat([subj_boxes[:, :, 0::2], obj_boxes[:, :, 0::2]], dim=-1)  # x1, x2
        all_y = torch.cat([subj_boxes[:, :, 1::2], obj_boxes[:, :, 1::2]], dim=-1)  # y1, y2

        all_x, _ = all_x.sort(dim=-1)
        all_y, _ = all_y.sort(dim=-1)

        # return [x1, y1, x2, y2] between boxes
        return torch.stack([all_x[:, :, 1], all_y[:, :, 1], all_x[:, :, 2], all_y[:, :, 2]], dim=-1)
    
    @classmethod
    def get_enclosing_action_boxes(cls, boxes):
        """
        Compute enclosing action boxes.

        Args:
            boxes (torch.Tensor): Tensor of shape [B, N, 2, 4], where the last dimension 
                                is [x1, y1, x2, y2] and dim=2 indexes subject and object.

        Returns:
            torch.Tensor: Union action boxes of shape [B, N, 4]
        """
        subj_boxes = boxes[:, :, 0, :]  # [B, N, 4]
        obj_boxes = boxes[:, :, 1, :]   # [B, N, 4]

        x1 = torch.min(subj_boxes[:, :, 0], obj_boxes[:, :, 0])
        y1 = torch.min(subj_boxes[:, :, 1], obj_boxes[:, :, 1])
        x2 = torch.max(subj_boxes[:, :, 2], obj_boxes[:, :, 2])
        y2 = torch.max(subj_boxes[:, :, 3], obj_boxes[:, :, 3])

        return torch.stack([x1, y1, x2, y2], dim=-1)

class GroundingEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, hidden_size=512, text_encoder_dim=4096,
                 max_hoi_seq=32, 
                 pos_embed_dim=32, role_embed_dim=32,
                 fourier_freq=32, init_logit=-5.0,
                 n_roles=3, role_std=0.02, mlp_out_std=3e-4):
        super(GroundingEncoder, self).__init__()
        self.text_encoder_dim = text_encoder_dim
        
        self.role_emb = nn.Embedding(n_roles, role_embed_dim)
        nn.init.kaiming_normal_(self.role_emb.weight, nonlinearity="linear")
        
        self.pos_embed_dim = pos_embed_dim
        self.max_hoi_seq = max_hoi_seq
        self.fourier_freq = fourier_freq
        self.fourier_dim = fourier_freq * 2 * 4
        
        position = torch.arange(max_hoi_seq).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_embed_dim, 2) * (-math.log(10000.0) / pos_embed_dim))
        pe = torch.zeros(max_hoi_seq, pos_embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
        self.norm = nn.LayerNorm(text_encoder_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(text_encoder_dim + self.fourier_dim + pos_embed_dim + role_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, text_encoder_dim),
        )
        
        self.gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x, role_ids: torch.Tensor, idx_ids: torch.Tensor, boxes: torch.Tensor):
        # x = prompt_embeds: [B, T, D]
        # role_ids: [hoi_seq_len]
        # idx_ids: [hoi_seq_len]
        B, _, _ = x.shape
        h = self.norm(x)
        
        role_pe = self.role_emb(role_ids)
        role_pe = role_pe.unsqueeze(0).expand(B, -1, -1)
        
        if self.training:
            idx_ids = (idx_ids + torch.randint(0, self.max_hoi_seq, (B, 1), device=x.device) ) % self.max_hoi_seq
            idx_pe = self.pe[idx_ids]
        else:
            idx_pe = self.pe[idx_ids].unsqueeze(0).expand(B, -1, -1)

        idx_pe = idx_pe - idx_pe.mean(-1, keepdim=True)
        boxes_features = get_fourier_embeds_from_boundingbox(self.fourier_freq, boxes) # [B, T, D=256]

        h = torch.cat([h, boxes_features, idx_pe, role_pe], dim=-1)
        x = x + self.gate.tanh() * self.mlp(h)  # [B, T, D=4096]

        return x


if __name__ == "__main__":
    # Example usage
    boxes = [[[0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]], 
             [[0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6],
              [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6],
              [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6],
              [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]],]
    box_labels = [["person", "dog"],
                  ["person", "cat", "person", "dog",
                   "person", "cat", "person", "dog",
                   "person", "cat", "person", "dog",
                   "person", "cat", "person", "dog",]]
    hoi_labels = [["walking"],
                  ["walking", "running",
                   "walking", "running",
                   "walking", "running",
                   "walking", "running",]]
    
    tokenizer = T5TokenizerFast.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="tokenizer_2")
    text_encoder = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", subfolder="text_encoder_2")

    # processed_boxes, box_prompt_embeds = GroundingInput.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder)
    
    # # print(processed_boxes)
    # print(f"processed_boxes: {len(processed_boxes)}x{len(processed_boxes[0])}x{len(processed_boxes[0][0])}x4")
    # print(f"processed_boxes: {len(processed_boxes)}x{len(processed_boxes[1])}x{len(processed_boxes[1][0])}xD")
    # # print(box_prompt_embeds)
    # print(f"box_prompt_embeds: {len(box_prompt_embeds)}x{len(box_prompt_embeds[0])}x{len(box_prompt_embeds[0][0])}xD")
    # print(f"box_prompt_embeds: {len(box_prompt_embeds)}x{len(box_prompt_embeds[1])}x{len(box_prompt_embeds[1][0])}xD")
    
    # # Prepare HOI input
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_hoi(box_prompt_embeds, processed_boxes)
    # print(f"out_embeds: {out_embeds.shape}")  # [B, N_max * M * T_max, D]
    # print(f"out_boxes: {out_boxes.shape}")    # [B, N_max * M * T_max, 4]
    # print(f"out_text_ids: {out_text_ids.shape}")  # [B, N_max * M * T_max, 3]
    # # Vanilla T2I
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_pipeline_input(tokenizer, text_encoder,
    #                                                                             boxes=None, box_labels=None, hoi_labels=None)
    # print(f"out_embeds: {out_embeds}")  # None
    # print(f"out_boxes: {out_boxes}")    # None
    # print(f"out_text_ids: {out_text_ids}")  # None
    # # HOI text control
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_pipeline_input(tokenizer, text_encoder,
    #                                                                             boxes=None, box_labels=box_labels, hoi_labels=hoi_labels)
    # print(f"out_embeds: {out_embeds.shape}")
    # print(f"out_boxes: {out_boxes.shape}")
    # print(f"out_text_ids: {out_text_ids.shape}")
    # # HOI box control
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_pipeline_input(tokenizer, text_encoder,
    #                                                                             boxes=boxes, box_labels=box_labels, hoi_labels=hoi_labels)
    # print(f"out_embeds: {out_embeds.shape}")
    # print(f"out_boxes: {out_boxes.shape}")    # None
    # print(f"out_text_ids: {out_text_ids.shape}")  # None
    
    # # Box control with no HOI labels
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_pipeline_input(tokenizer, text_encoder,
    #                                                                             boxes=boxes, box_labels=box_labels, hoi_labels=None)
    # print(f"out_embeds: {out_embeds.shape}")  # [B, N_max * M * T_max, D]
    # print(f"out_boxes: {out_boxes.shape}")    # [B, N_max * M * T_max, 4]
    # print(f"out_text_ids: {out_text_ids.shape}")  # [B, N_max * M * T_max, 3]
    
    # # Box control with no boxes
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_pipeline_input(tokenizer, text_encoder,
    #                                                                             boxes=None, box_labels=box_labels, hoi_labels=None)
    # print(f"out_embeds: {out_embeds.shape}")  # [B, N_max * M * T_max, D]
    # print(f"out_boxes: {out_boxes.shape}")    # [B, N_max * M * T_max, 4]
    # print(f"out_text_ids: {out_text_ids.shape}")  # [B, N_max * M * T_max, 3]
    
    #### NEW TEST ####
    
    # processed_boxes, box_prompt_embeds = GroundingInput.preprocess_box(boxes, box_labels, tokenizer, text_encoder)
    # print(f"processed_boxes: {len(processed_boxes)}x{len(processed_boxes[0])}x{len(processed_boxes[0][0])}x4")
    # print(f"box_prompt_embeds: {len(box_prompt_embeds)}x{len(box_prompt_embeds[0])}x{len(box_prompt_embeds[0][0])}xD")  # D is the embedding dimension

    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_box(box_prompt_embeds, processed_boxes)
    # print(f"out_embeds: {out_embeds.shape}")  # [B, N_max * M * T_max, D]
    # print(f"out_boxes: {out_boxes.shape}")    # [B, N_max * M * T_max, 4]
    # print(f"out_text_ids: {out_text_ids.shape}")  # [B, N_max * M * T_max, 3]
    
    processed_boxes, box_prompt_embeds = GroundingInput.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder)
    out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_hoi(box_prompt_embeds, processed_boxes)
    
    attn_mask, _ = GroundingInput.prepare_attention_mask(out_text_ids, out_boxes, use_union_action_mask=True)
    pass
    
    #### TEST RANDOM DROP HOI ####
    # from datasets import load_from_disk
    
    # # ds = load_from_disk("data/synthesis_edits_kontext_9")
    # ds = load_from_disk("data/hicodet_kontext_dataset")
    # ds.set_format(type='torch', columns=['hois', 'boxes'])
    # # sample = ds[0::512] # batched for hoi edits datasets
    # sample = ds[0::16000] # batched for hicodet_kontext_dataset
    # boxes = sample['boxes']
    # hois = sample['hois']
    # objects = [None]
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_train_input(tokenizer, text_encoder,
    #                                                                          boxes=boxes, hois=hois, objects=objects,
    #                                                                          random_drop_boxes=0, random_drop_hois=1)
    # for b in range(len(boxes)):
    #     n = len(boxes[b])
    #     assert (boxes[b] == out_boxes[b, ::64][:n*2].reshape(boxes[b].shape)).all()  # boxes are [B, N, 1, 4], out_boxes is [B, T_max, 4]
    
    #### TEST LIMIT MAX HOI ####
    # boxes = [[[0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]], 
    #          [[0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6],
    #           [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6],
    #           [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6], [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]],]
    # box_labels = [["person", "dog"],
    #               ["person", "cat", "person", "dog",
    #                "person", "cat", "person", "dog",
    #                "person", "cat", "person", "dog",]]
    # hoi_labels = [["walking"],
    #               ["walking", "running",
    #                "walking", "running",
    #                "walking", "running",]]
    
    # processed_boxes, box_prompt_embeds = GroundingInput.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder)
    # assert len(processed_boxes[1]) == HOI_N_MAX

    # #### TEST LIMIT MAX BOX ####
    # processed_boxes, box_prompt_embeds = GroundingInput.preprocess_box(boxes, box_labels, tokenizer, text_encoder)
    # assert len(processed_boxes[1]) == BOX_N_MAX
    
    # #### TEST FIXED EMBED for HOI ROLE ####
    # processed_boxes, box_prompt_embeds = GroundingInput.preprocess_hoi(boxes, box_labels, hoi_labels, tokenizer, text_encoder)
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_hoi(box_prompt_embeds, processed_boxes)
    
    # grounding_encoder = GroundingEncoder()
    # mlp_prompt_embeds = grounding_encoder(out_embeds, out_text_ids[:, 2])
    
    #### TEST FIXED EMBED for BOX ROLE ####
    # processed_boxes, box_prompt_embeds = GroundingInput.preprocess_box(boxes, box_labels, tokenizer, text_encoder)
    # out_embeds, out_boxes, out_text_ids = GroundingInput.prepare_box(box_prompt_embeds, processed_boxes)
    # new_text_ids = GroundingInput.get_rope_ids(out_text_ids, img_width=64, img_height=48, cond_width=64, cond_height=48)

    # grounding_encoder = GroundingEncoder()
    # grounding_encoder.eval()
    # mlp_prompt_embeds = grounding_encoder(out_embeds, out_text_ids[:, 2], out_text_ids[:, 1])
    
    # grounding_encoder.train()
    # mlp_prompt_embeds = grounding_encoder(out_embeds, out_text_ids[:, 2], out_text_ids[:, 1])
    
    grounding_encoder = GroundingEncoder()
    grounding_encoder.eval()
    mlp_prompt_embeds = grounding_encoder(out_embeds, out_text_ids[:, 2], out_text_ids[:, 1], boxes=out_boxes)
    
    # mix_boxes: [B, N, M, 4]
    # mix_boxes = [
    #     [ # b=0
    #         [ # n=0
    #             [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]
    #         ],
    #         [ # n=1
    #             None
    #         ],
    #     ],
    #     [ # b=1
    #         [ # n=0
    #             None, None
    #         ],
    #         [ # n=1
    #             [0, 0, 0.2, 0.2]
    #         ],
    #     ]
    # ]
    # # mix_box_labels: [B, N, M]
    # mix_box_labels = [
    #     [ # b=0
    #         ["person", "dog"], # n=0
    #         ["cat"] # n=1
    #     ],
    #     [ # b=1
    #         ["person", "dog"],
    #         ["watermelon"]
    #     ]
    # ]
    # # mix_box_labels: [B, N]
    # mix_hoi_labels = [
    #     [ # b=0
    #         "walking", None,
    #     ],
    #     [ # b=1
    #         "hugging", None,
    #     ],
    # ]
    # mix boxes: [B=1, N, M, 4]
    mix_boxes = [
        [ # b=0
            [ # n=0
                [0, 0, 0.2, 0.2], [0.4, 0.4, 0.6, 0.6]
            ],
            [ # n=1
                None
            ],
        ]
    ]
    # mix_box_labels: [B, N, M]
    mix_box_labels = [
        [ # b=0
            ["person", "dog"], # n=0
            ["cat"] # n=1
        ]
    ]
    # mix_box_labels: [B, N]
    mix_hoi_labels = [
        [ # b=0
            "walking", None,
        ]
    ]
    GroundingInput.prepare_mixed_pipeline_input(tokenizer=tokenizer, text_encoder=text_encoder,
                                                mix_boxes=mix_boxes, mix_box_labels=mix_box_labels, mix_hoi_labels=mix_hoi_labels,
                                                hoi_seq_len=64, max_box=9)
    pass