import argparse
import json
import os
import random
import re
import time
import warnings
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import torch
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
# from diffusers import FluxFillPipeline
from groundingdino.models import build_model
from groundingdino.util.inference import (annotate, load_image, load_model,
                                          predict)
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.ops import box_convert
from tqdm import tqdm
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import ast


import json
import os
import fcntl

# Set up logger
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MAIR_LAB_DATA_DIR = '/mnt/localssd'
VIM_DATA_DIR = '/mnt/localssd/vim_data'
dataset = 'coco'
diffusion_model_name = 'sdxl'
split = 'train'

output_dir_root = os.path.join(VIM_DATA_DIR, f"{dataset}_{diffusion_model_name}_edited_{split}")

def set_random_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_coco_path_by_image_id(split: str, image_id: Union[str, int]):
    image_id_str = f"{int(image_id):012d}"
    return os.path.join(MAIR_LAB_DATA_DIR, "coco", "images", f"{split}2017", f"{image_id_str}.jpg")


def get_output_dir_path_by_image_id(output_dir: str, image_id):
    base_path = os.path.join(output_dir, f"{image_id}")
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    return base_path

def load_model_hf( repo_id, filename, ckpt_config_filename, device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location="cpu")
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    logger.info(f"Model loaded from {cache_file} \n => {log}")
    return model

def generate_grounding(dino_model, image, text_prompt, box_threshold=0.35, text_threshold=0.25):
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cpu",
    )

    logger.info(f"no of boxes: {len(boxes)}, phrases: {phrases}, boxes: {boxes}")

    # check or remove boxes that capture the entire image
    boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    boxes_area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])

    # Keep only the boxes that cover less than 85% of the image
    boxes_area_tensor = boxes_area
    mask = boxes_area_tensor < 0.85
    indices = np.where(mask)[0].tolist()
    #boxes_xyxy = boxes_xyxy[mask]  # making sure boxes is a tensor
    #phrases = [phrases[i] for i in indices]

    logger.info(f"no of boxes (post-filtering): {len(boxes_xyxy)}, phrases: {phrases}, boxes: {boxes_xyxy}")
    return boxes_xyxy, phrases
    
def generate_masks_with_grounding(image: PIL.Image.Image, boxes_xyxy):
    image_np = np.array(image)
    h, w, _ = image_np.shape
    boxes_unnorm = boxes_xyxy * np.array([w, h, w, h])
    logger.debug(f"boxes: {boxes_xyxy} => boxes_unnorm: {boxes_unnorm}")

    mask = np.zeros_like(image_np)
    for box in boxes_unnorm:
        x0, y0, x1, y1 = box
        mask[int(y0) : int(y1), int(x0) : int(x1), :] = 255
    return mask

def image_diffusion_edit_and_rank( image_id: str, image_path: str, input_caption: str, edits_info: List[Dict[str, str]]
                                  , device=None
):
    """
    image: an input image for diffusion
    edits_info: a list of dictionaries, each containing an edit instruction and a new caption
    return:
        outputs: a list of dictionaries, each containing an edit instruction, a new caption, and a highest-scored generated image
    """
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    dino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)

    start_time = time.time()
    num_images_per_prompt = 3  # number of sample images for diffusion
    negative_prompt = (
        "cropped, clipped, invisible, half visible, trimmed, distorted, (unnatural, unreal, unusual), (deformed iris, deformed pupils, "
        "semi-realistic, cgi, 3d, cartoon, anime), (deformed, distorted, disfigured), bad anatomy, "
        "extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, bad quality"
    )

    prompt_list = [f"{info['input_phrase']} => {info['edited_phrase']}" for info in edits_info]
    formatted_prompt_list = "\n".join(f"- {item}" for item in prompt_list[-20:])
    logger.info(f"Sampled prompt list (out of {len(prompt_list)}) :\n{formatted_prompt_list}")

    input_image, transformed_image = load_image(image_path)

    print(f'before grounding ..')
    image_data_list = [] 
    # Generate grounding and prepare image data
    for info in edits_info:
        try:
            grounding_phrase = info["input_phrase"]
            print(f'grounding_phrase   {grounding_phrase}')
            boxes, phrases = generate_grounding(
                dino_model, transformed_image, grounding_phrase, box_threshold=0.25, text_threshold=0.25
            )
            # if boxes is None or boxes.shape[0] == 0:
            #     boxes, phrases = self.diffuser.generate_grounding_kosmos2(input_image, grounding_phrase)
        except Exception as e:
            logger.error(f"Error in generate_grounding: {e}")

        # if boxes is None or boxes.shape[0] == 0:
        #     continue
        print(f'input_image: {input_image.shape}')
        mask_image = generate_masks_with_grounding(input_image, boxes)
        print(f'mask_image: {mask_image.shape}')
        image_data_list.append(
            {
                "mask_image": Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image,
                "boxes": boxes,
                "edit_info": info,
            }
        )
        mask_image = image_data_list[0]["mask_image"]
        print(f'mask_image: {mask_image.size}')
        # Ensure both images are same size
        input_size = input_image.size
        print(f'input_size: {input_size}')
        #mask_image = mask_image.resize(input_size)
        print(f'--> mask_image: {mask_image.size}')
        
        # Create new image with double width to hold both images
        combined_image = Image.new('RGB', (input_size[0] * 2, input_size[1]))
        
        # Paste input image and mask side by side
        combined_image.paste(input_image, (0, 0))
        combined_image.paste(mask_image, (input_size[0], 0))
        
        # Save combined image
        output_path = os.path.join(output_dir_root, f"input_and_mask_{image_id}.png")
        combined_image.save(output_path)
        logger.info(f"Saved combined input and mask image to {output_path}")

    input_image = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image
    # Save input image and mask side by side
    
        

    
device = "cuda" if torch.cuda.is_available() else "cpu"
set_random_seed()
os.environ["TOKENIZERS_PARALLELISM"] = "false"



with open('/mnt/localssd/coco_original_data.json', 'r', encoding='utf-8') as f:
    coco_original_data = json.load(f)
with open('/mnt/localssd/edited_object_data.json', 'r', encoding='utf-8') as f:
    edited_object_data = json.load(f)

#print(coco_original_data)
#print(edited_object_data)
# TODO: remove this 
tot_processed, success_count = 0, 0
# for idx, entry in tqdm(enumerate(annotations), desc="Editing images"):
#     image_id = entry["image_id"]
#     caption_text = entry["caption"]  # let's treat caption as prompt for stable-diffuser
#     if "image_path" in entry:
#         image_path = entry["image_path"]
#     else:
#         coco_split = "val" if self.split == "validation" else self.split
#         image_path = get_coco_path_by_image_id(split=coco_split, image_id=image_id)
#     logger.info(f"Processing image id: {image_id}, image caption: {caption_text}")
#     output_dir = self.get_output_dir_path_by_image_id(output_dir_root, image_id)
for idx, source_image_id in tqdm(enumerate(edited_object_data.keys()), desc="Editing images") :
    print(f'source_image_id: {source_image_id}')
    #if sample['category'] == 'object':
    edited_obj_sample = edited_object_data[source_image_id]
    print(f'edited_obj_sample: {edited_obj_sample}')
    image_id = edited_obj_sample['source_image_id']
    caption_text = coco_original_data[image_id]['caption']
    coco_split = "train"
    image_path = get_coco_path_by_image_id(split=coco_split, image_id=image_id)
    logger.info(f"Processing image id: {image_id}, image caption: {caption_text}")
    output_dir = get_output_dir_path_by_image_id(output_dir_root, image_id)

    # if self.dataset == "coco" and len([f for f in os.listdir(output_dir) if f.endswith(".png")]) >= 2:
    #     logger.info(f"Skipping image id: {image_id} as 2 png files already exist.")
    #     continue

    annotation_file_path = os.path.join(output_dir, "annotations.json")

    if True:
        # with self.file_locker.locked(output_dir) as lock_acquired:
        #     if not lock_acquired:
        #         logger.warning(f"Skipping image id: {image_id} as another process is working on it.")
        #         continue

        # edits_info = self.get_edit_instruction(image_id, caption_text, annotation_file_path)
        # if not edits_info:
        #     continue
        # edits_info = ds['train'][0]
        edited_object_data[image_id]
        edits_info = {}
        list_rep = ast.literal_eval(edited_obj_sample['edit_instruction'])
        edits_info['input_phrase'] = list_rep[0]
        edits_info['edited_phrase'] = list_rep[1]
        edits_info['edited_caption'] = edited_obj_sample['caption']
        edits_info['edit_id'] = edited_obj_sample['image_id']

        print(f'edits_info: {edits_info}')

        logger.info(f"Processing IDX # {idx}")
        logger.info(f"Original: {caption_text}, Edited: {edits_info}")

        edited_image_list = image_diffusion_edit_and_rank(image_id, image_path, caption_text, [edits_info], device)
        break