import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
import math
from datetime import timedelta
import numpy as np
import decord
from decord import VideoReader
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import os
import json
from multiprocessing import Pool
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm
import datetime
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video Captioning using Qwen2.5-VL')
    parser.add_argument('--input_dir', type=str, default="IntentVC/",
                        help='Root directory containing video folders')
    parser.add_argument('--model_name', type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct", help='Model name or path')
    parser.add_argument('--use_flash_attn',
                        action='store_true', help='Use Flash Attention 2')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save the captions')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use for parallel processing')
    parser.add_argument('--public_sample', type=str, default="sample_result_public.json",
                        help='Path to sample_result_public.json')
    parser.add_argument('--private_sample', type=str, default="sample_result_private.json",
                        help='Path to sample_result_private.json')
    parser.add_argument('--train_data', type=str, default="train.json",
                        help='Path to training data file')
    return parser.parse_args()


def read_bbox_file(bbox_path):
    """Read bounding box information from the text file.
    Format: x, y, w, h
    """
    try:
        with open(bbox_path, 'r') as f:
            bboxes = [list(map(int, line.strip().split(','))) for line in f]
        return bboxes
    except Exception as e:
        print(f"Error reading bbox file {bbox_path}: {str(e)}")
        return []


def get_video_ids_from_sample(sample_path):
    """Get video IDs from sample result file."""
    try:
        with open(sample_path, 'r') as f:
            data = json.load(f)
            return list(data['captions'].keys())
    except Exception as e:
        print(f"Error reading sample file {sample_path}: {str(e)}")
        return []


def get_video_paths(input_dir, video_ids):
    """Get all video paths and their corresponding bbox files for given video IDs."""
    video_paths = []
    for video_id in video_ids:
        # Extract category and number from video_id (e.g., "airplane-1" -> "airplane", "1")
        category, number = video_id.split('-')
        video_dir = os.path.join(input_dir, category, f"{category}-{number}")
        video_path = os.path.join(video_dir, f"{category}-{number}.mp4")
        bbox_path = os.path.join(video_dir, "object_bboxes.txt")
        
        if os.path.exists(video_path) and os.path.exists(bbox_path):
            video_paths.append((video_id, video_path, bbox_path))
        else:
            print(f"Warning: Missing files for {video_id}")
    
    return video_paths

def resize_image_with_aspect_ratio(image, max_size=1024):
    """Resize image keeping aspect ratio so that the longest edge is max_size."""
    width, height = image.size
    if max(width, height) <= max_size:
        return image

    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_train_data(train_path):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    train_data = train_data['captions']
    object_captions = {}
    for key, value in train_data.items():
        obj_key = key.split('-')[0]
        if obj_key not in object_captions:
            object_captions[obj_key] = []
        object_captions[obj_key].append(value)
    return object_captions


def generate_caption_for_segment(model, processor, video_path, bbox_path, max_new_tokens=512, train_captions=None):
    """Generate caption for a video with its bounding boxes."""
    # try:
    if True:
        # Read video frames
        vr = VideoReader(video_path)
        total_frames = len(vr)
        
        # Get object category from video path
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        object_category = video_id.split('-')[0]
        
        # Get example captions for this object category
        example_captions = ""
        if train_captions and object_category in train_captions:
            # Get up to 3 random examples
            examples = train_captions[object_category]
            if len(examples) > 3:
                import random
                examples = random.sample(examples, 3)
            example_captions = "\nExample captions for this object category:\n" + "\n".join([f"- {cap}" for cap in examples])
        
        # Read bounding boxes
        bboxes = read_bbox_file(bbox_path)
        
        # Create directory for cropped video
        cropped_dir = os.path.join('cropped_videos', video_id)
        os.makedirs(cropped_dir, exist_ok=True)
        
        # Process all frames
        cropped_frames = []  # List to store cropped frames
        
        # Process frames in batches to avoid memory issues
        batch_size = 32
        for i in range(0, total_frames, batch_size):
            end_idx = min(i + batch_size, total_frames)
            frame_indices = list(range(i, end_idx))
            frames = vr.get_batch(frame_indices).asnumpy()
            
            for frame_idx, frame in enumerate(frames):
                original_frame_idx = frame_indices[frame_idx]
                if original_frame_idx < len(bboxes):
                    x, y, w, h = bboxes[original_frame_idx]
                    
                    # Convert frame to PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # Crop the region inside the bounding box
                    cropped_region = pil_image.crop((x, y, x + w, y + h))
                    cropped_frames.append(np.array(cropped_region))
                else:
                    print(f"Warning: No bounding box found for frame {original_frame_idx}")
        
        # Save cropped video if we have frames
        if cropped_frames:
            # Get video properties from original video
            fps = vr.get_avg_fps()
            height, width = cropped_frames[0].shape[:2]
            
            # Create video writer using decord
            cropped_video_path = os.path.join(cropped_dir, f"{video_id}_cropped.mp4")
            
            # Convert frames to numpy array
            frames_array = np.stack(cropped_frames)
            
            # Save video using decord
            writer = decord.VideoWriter(cropped_video_path, width, height, fps)
            writer.write(frames_array)
            writer.close()
            
            # Verify the video duration
            reader = decord.VideoReader(cropped_video_path)
            frame_count = len(reader)
            actual_fps = reader.get_avg_fps()
            duration = frame_count / actual_fps
            reader.close()
            
            print(f"Saved cropped video to {cropped_video_path}")
            print(f"Video duration: {duration:.2f} seconds")
            print(f"Frame count: {frame_count}")
            print(f"FPS: {actual_fps}")
            
        # Calculate frame indices for uniform sampling of 10 frames for captioning
        frame_indices = np.linspace(0, total_frames-1, 10, dtype=int)
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Create debug directory for saving frames
        debug_dir = os.path.join('debug_frames', video_id)
        os.makedirs(debug_dir, exist_ok=True)
        
        # Process frames and prepare frame list for captioning
        frame_list = []
        for frame_idx, frame in enumerate(frames):
            pil_image = Image.fromarray(frame)
            pil_image = resize_image_with_aspect_ratio(pil_image, max_size=480)
            
            # Add bounding box to the image if available
            original_frame_idx = frame_indices[frame_idx]
            if original_frame_idx < len(bboxes):
                x, y, w, h = bboxes[original_frame_idx] 
                # Convert bbox coordinates to the resized image scale
                width, height = pil_image.size
                scale_x = width / frame.shape[1]
                scale_y = height / frame.shape[0]
                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)
                
                # Draw bounding box on the image
                draw = ImageDraw.Draw(pil_image)
                draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                
                # Save the frame with bounding box
                frame_path = os.path.join(debug_dir, f'frame_{original_frame_idx:04d}.jpg')
                pil_image.save(frame_path)
            else:
                print(f"Warning: No bounding box found for frame {original_frame_idx}")
            
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            frame_list.append(f"data:image/jpeg;base64,{img_str}")

        # Prepare bbox information for the prompt
        # bbox_info = json.dumps(bboxes, indent=2)

        # messages = [
        #     {
        #         "role": "system",
        #         "content":
        #          "You are an expert video captioning model. Your task is to generate objective captions focusing on the object highlighted in the red bounding box. "
        #          "Describe only what is visible within the red box, including the object's appearance, motion, interactions, and spatial context. "
        #          "Use a single, concise, third-person, present-tense declarative sentence. "
        #          "Do not mention any objects or actions outside the red box."
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": frame_list,
        #                 "fps": 1.0,
        #             },
        #             {
        #                 "type": "text",
        #                 "text": (
        #                     f"Generate one English caption that objectively describes the visual content of a video clip, focusing only on the object highlighted in the red bounding box. "
        #                     f"The bounding box information for each frame is provided below:\n\n{bbox_info}\n\n"
        #                     "The caption should:\n"
        #                     "1. Clearly describe the object's appearance\n"
        #                     "2. Describe its motion and movement\n"
        #                     "3. Include any interactions with the environment\n"
        #                     "4. Provide spatial context\n\n"
        #                     f"{example_captions}\n\n"
        #                     "Format your response as a JSON object with a single key 'object' and a detailed caption as the value.\n"
        #                     "Example format:\n"
        #                     "{\n"
        #                     '  "object": "a small white airplane descends towards a runway amid the mountainous terrain"\n'
        #                     "}\n\n"
        #                     "Remember to use a single, concise, third-person, present-tense declarative sentence."
        #                 )
        #             }
        #         ]
        #     }
        # ]
        messages = [
            {
                "role": "system",
                "content":
                    "You are a professional video captioning model, specialized in generating objective and concise descriptions for short video clips. "
                    "Your sole task is to describe **only the object** inside the red bounding box across the video frames. "
                    "Ignore any visual elements or actions that are outside the bounding box. "
                    "Your caption must be a single, third-person, present-tense declarative sentence. "
                    "Avoid speculation, subjective language, or information not visually supported by the red box."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_list,
                        "fps": 1.0,
                    },
                    {
                        "type": "text",
                        "text": (
                            "Generate one English caption that objectively describes the visual content **strictly inside the red bounding box** over the given video clip. "
                            "Your caption should fulfill the following requirements:\n"
                            "1. Clearly describe the object's **appearance** (e.g., color, size, shape, structure)\n"
                            "2. Describe its **motion** or dynamic behavior (e.g., flies, runs, hovers, spins)\n"
                            "3. Mention **interactions** with objects **only if they are inside** the red box\n"
                            "4. Include **spatial context** only if it can be inferred from within the red box\n\n"
                            "Do NOT:\n"
                            "- Mention any entities or actions outside the red box\n"
                            "- Use subjective or speculative terms (e.g., 'appears to be', 'seems like')\n"
                            "- Generate more than one sentence\n"
                            "- Use first-person or second-person phrasing\n\n"
                            "Here are some reference-style captions for consistency:\n"
                            f"{example_captions}\n\n"
                            "Format your response strictly as a JSON object with a single key 'object' and the caption string as the value. For example:\n"
                            "{\n"
                            '  "object": "a small white airplane descends towards a runway amid the mountainous terrain"\n'
                            "}\n\n"
                            "Now generate the caption based on the input video and red bounding box annotations."
                        )
                    }
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return text, caption
    # except Exception as e:
    #     print(f"\nError processing video {video_path}: {str(e)}")
    #     return None


def process_video(video_path, bbox_path, model, processor, max_new_tokens, train_captions=None):
    """Process a single video and return its captions."""
    # try:
    if True:
        prompt, caption = generate_caption_for_segment(
            model,
            processor,
            video_path,
            bbox_path,
            max_new_tokens=max_new_tokens,
            train_captions=train_captions
        )
        
        if caption is None:
            raise ValueError("Failed to generate caption")
            
        return prompt, caption
    # except Exception as e:
    #     print(f"\nError processing entire video {video_path}: {str(e)}")
    #     return None


def split_list(lst, n):
    """Split a list into n roughly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_videos_on_gpu(process_args, args):
    """Process a batch of videos on a single GPU."""
    gpu_id, video_triples = process_args
    torch.cuda.set_device(gpu_id)
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": f"cuda:{gpu_id}"
    }
    if args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    
    # Load training data for example captions
    train_captions = load_train_data(args.train_data)
    
    # Create temp file for intermediate results
    temp_file = os.path.join(args.output_dir, "temp_captions.jsonl")
    
    results = {}
    for video_id, video_path, bbox_path in tqdm(video_triples, desc=f"GPU {gpu_id}", position=gpu_id):
        # try:
        if True:
            # Get the prompt and caption
            prompt, caption = process_video(
                video_path,
                bbox_path,
                model,
                processor,
                args.max_new_tokens,
                train_captions
            )
            
            # Clean up caption by removing markdown code block markers
            if caption:
                caption = caption.replace('```json\n', '').replace('\n```', '')
            
            # Save intermediate caption and prompt to JSONL file
            with open(temp_file, 'a', encoding='utf-8') as f:
                json.dump({
                    'video_id': video_id,
                    'prompt': prompt,
                    'caption': str(caption),
                    'timestamp': str(datetime.datetime.now())
                }, f, ensure_ascii=False)
                f.write('\n')
            
            if caption:
                # Try to parse the caption as JSON
                try:
                    caption_dict = json.loads(caption)
                    results[video_id] = caption_dict.get('object', '')
                except json.JSONDecodeError:
                    # If not valid JSON, use the caption string directly
                    results[video_id] = caption.strip()
            else:
                results[video_id] = ''
                
        # except Exception as e:
        #     print(f"\nError processing video {video_path} on GPU {gpu_id}: {str(e)}")
        #     results[video_id] = ''
    
    del model
    del processor
    torch.cuda.empty_cache()
    
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get video IDs from sample files
    public_video_ids = get_video_ids_from_sample(args.public_sample)
    private_video_ids = get_video_ids_from_sample(args.private_sample)
    
    # Get video paths for both public and private sets
    public_video_paths = get_video_paths(args.input_dir, public_video_ids)
    private_video_paths = get_video_paths(args.input_dir, private_video_ids)
    
    if not public_video_paths and not private_video_paths:
        print("Error: No valid video-bbox pairs found")
        return
    
    # Process public and private sets separately
    for video_paths, output_name, original_ids in [(public_video_paths, "result_public.json", public_video_ids), 
                                   (private_video_paths, "result_private.json", private_video_ids)]:
        if not video_paths:
            continue
            
        # Remove random shuffle to maintain order
        num_gpus = min(args.num_gpus, torch.cuda.device_count())
        if num_gpus < 1:
            print("Error: No GPUs available")
            return
        
        video_groups = split_list(video_paths, num_gpus)
        process_args = [(i, group) for i, group in enumerate(video_groups)]
        
        mp.set_start_method('spawn', force=True)
        
        print(f"Starting video processing on {num_gpus} GPUs for {output_name}...")
        with Pool(num_gpus) as pool:
            all_results_groups = pool.map(partial(process_videos_on_gpu, args=args), process_args)
        
        # Combine results
        combined_results = {}
        for results_group in all_results_groups:
            combined_results.update(results_group)
        
        # Reorder results according to original video_ids
        ordered_results = OrderedDict()
        for video_id in original_ids:
            if video_id in combined_results:
                ordered_results[video_id] = combined_results[video_id]
            else:
                ordered_results[video_id] = ""  # Handle missing results
        
        # Save results in the required format
        output_data = {
            "version": "Submission File Example VERSION 1.0",
            "captions": ordered_results
        }
        
        output_file = os.path.join(args.output_dir, output_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
