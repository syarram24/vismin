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
#from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
from diffusers import FluxFillPipeline

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

def sd_masked_inpainting(
    pipe,
    prompt: Union[str, List[str]] = None,
    image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_inference_steps: int = None,
    guidance_scale: float = None,
    num_images_per_prompt: int = None,
    strength: float = None,
):
    image_inpainting = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        strength=strength,
        output_type="pil",
    ).images

    return image_inpainting

def image_diffusion_edit_and_rank( image_id: str, image_path: str, input_caption: str, edits_info: List[Dict[str, str]]
                                  , device=None
):
    """
    image: an input image for diffusion
    edits_info: a list of dictionaries, each containing an edit instruction and a new caption
    return:
        outputs: a list of dictionaries, each containing an edit instruction, a new caption, and a highest-scored generated image
    """
    pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, use_auth_token=True).to(device)
    # Check if PyTorch version is 2.x

    # load grounding model
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
        width, height, _ = input_image.shape
  
        # Convert PIL images to numpy arrays if needed
        input_np = np.array(input_image) if isinstance(input_image, PIL.Image.Image) else input_image
        mask_np = np.array(mask_image) if isinstance(mask_image, PIL.Image.Image) else mask_image

        # Create a combined image by placing input and mask side by side
        combined_image = np.hstack((input_np, mask_np))

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the combined image
        output_path = os.path.join(output_dir, f"input_and_mask.png")
        cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved combined input and mask image to {output_path}")

    



    input_image = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image

    if not image_data_list:
        logger.warning(f"Could not generate grounding for image id: {image_path}, caption: {input_caption}")
        return []

    # Image generation for each engine
    for data in image_data_list:
        info = data["edit_info"]
        config = {
            "scheduled_sampling_beta": random.choice([x / 10 for x in range(5, 11)]),
            "num_inference_steps": random.choice(range(50, 100, 5)),
            "guidance_scale": random.choice(np.arange(5, 12, 1.5).tolist()),
        }
        use_negative_prompt = random.choice([True, False])

    # Batch processing for engines like 'sdxl'
    batch_size = 1
    for i in range(0, len(image_data_list), batch_size):
        batch_data = image_data_list[i : i + batch_size]
        input_images = [input_image.resize((1024, 1024)) for _ in batch_data]
        mask_images = [data["mask_image"].resize((1024, 1024)) for data in batch_data]

        prompts = []
        for data in batch_data:
            edited_phrase = data["edit_info"]["edited_phrase"]
            edit_id = data["edit_info"]["edit_id"]
            # conditionally add enhanced phrases if they exist to the prompt
            # enhanced_phrases = self.generated_edit_enhanced_phrases.get(str(image_id), {}).get(edit_id, [])
            # if isinstance(enhanced_phrases, list) and all(isinstance(item, str) for item in enhanced_phrases):
            #     enhanced_phrases_str = ", ".join(enhanced_phrases[:3])
            #     logger.info(f"{edited_phrase} => enhanced phrases: {enhanced_phrases_str}")
            #     edited_phrase = edited_phrase + ", " + enhanced_phrases_str if enhanced_phrases else edited_phrase
            prompts.append(edited_phrase + ". " + data["edit_info"]["edited_caption"])
            strenth = 0.99

        print(f'prompts: {prompts}')
        REPEAT = 2
        generated_images_batch = []
        for _ in range(REPEAT):
            num_inference_steps = (
                random.choice(range(20, 50, 5))
                # if self.diffuser.diffusion_model_name == "sdxl"
                # else random.choice(range(35, 75, 5))
            )
            config = {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": random.choice(np.arange(5, 12, 1.5).tolist()),
                "strength": strenth,
            }
            use_negative_prompt = random.choice([True, False])
            print(device)
            print(f'input_images {len(input_images)} mask_images {len(mask_images)}')
            generated_images = pipe(
                prompt=prompts,
                image=input_images,
                mask_image=mask_images,
                height=1632,
                width=1232,
                guidance_scale=30,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(0)
                
            ).images
            #[
            # generated_images = sd_masked_inpainting(
            #     pipe=pipe,
            #     prompt=prompts,
            #     image=input_images,
            #     mask_image=mask_images,
            #     negative_prompt=[negative_prompt] * len(prompts) if use_negative_prompt else None,
            #     **config,
            #     num_images_per_prompt=num_images_per_prompt,
            # )
            generated_images_batch.extend(generated_images)
        # generated_images_batch = group_outputs_for_batch_repeats(
        #     generated_images_batch, batch_size, REPEAT, num_images_per_prompt
        # )

        # for data in batch_data:
        #     data.update(
        #         generated_images=generated_images_batch.pop(0),
        #         use_negative_prompt=use_negative_prompt,
        #         **config,
        #     )
        # assert len(generated_images_batch) == 0, "generated_images should be empty"

    
        

    
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
for idx, source_image_id in tqdm(enumerate(sorted(edited_object_data.keys())), desc="Editing images") :
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


# import torch
# from diffusers import FluxFillPipeline
# from diffusers.utils import load_image

# image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
# mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

# pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16).to("cuda")
# image = pipe(
#     prompt="a white paper cup",
#     image=image,
#     mask_image=mask,
#     height=1632,
#     width=1232,
#     guidance_scale=30,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save(f"flux-fill-dev.png")