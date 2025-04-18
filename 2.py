import os
import torch
import psutil
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
import torchvision.transforms as T
import imageio

# Configuration for models and directories
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SD_MODEL = "CompVis/stable-diffusion-tiny"  # Tiny-SD (small and optimized)
DEFAULT_SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid"  # ZeroScope for lightweight video generation
DEFAULT_LLM_MODEL = "huggingface/tinyllama-1.1B"  # Small LLM (TinyLlama)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Check if directories exist and are accessible
def validate_directory(path: Path, is_input=True):
    if not path.exists():
        raise FileNotFoundError(f"{'Input' if is_input else 'Output'} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{'Input' if is_input else 'Output'} path is not a directory: {path}")
    print(f"[✓] {'Input' if is_input else 'Output'} directory validated: {path}")

# Step 1: Enhance Prompt (using tinyllama-1.1B)
def enhance_prompt(prompt, model_name=DEFAULT_LLM_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)

    prompt_template = f"Enhance this image generation prompt for style, clarity, and visual richness: '{prompt}'"
    inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return result.split(":")[-1].strip()

# Step 2: Semantic Search (using all-MiniLM-L6-v2)
def find_references(prompt, asset_dir, top_k=3):
    embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, device=device)
    files = [f for f in asset_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    if not files:
        return []

    names = [f.stem.replace('_', ' ') for f in files]
    corpus_embeddings = embedder.encode(names, convert_to_tensor=True)
    query_embedding = embedder.encode(prompt, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_indices = torch.topk(scores, k=min(top_k, len(files))).indices
    return [files[i] for i in top_indices]

# Step 3: Image Generation (using Tiny-SD)
def generate_image(prompt, refs, model_name=DEFAULT_SD_MODEL):
    prompt += ", " + ", ".join([r.stem.replace('_', ' ') for r in refs])
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

# Step 4: Video Generation (using ZeroScope or Stable Video Diffusion)
def image_to_video(image: Image.Image, frames=14, fps=6, model_name=DEFAULT_SVD_MODEL):
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)

    pipe = StableVideoDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    video = pipe(image_tensor, decode_chunk_size=8, motion_bucket_id=127, noise_aug_strength=0.02).frames[0]
    return video

# Step 5: Save Video (optional)
def save_video(frames, output_dir, name="output.mp4", fps=6):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / name
    frame_list = [np.array(f.convert("RGB")) for f in frames]
    imageio.mimsave(out_path, frame_list, fps=fps)
    print(f"[✓] Saved video to: {out_path}")
    return out_path

# Main pipeline
def run_pipeline(user_prompt, input_dir, output_dir):
    # Validate directories
    validate_directory(input_dir, is_input=True)
    validate_directory(output_dir, is_input=False)

    print("[1] Enhancing prompt...")
    enhanced = enhance_prompt(user_prompt)
    print(f"[✓] Enhanced prompt: {enhanced}")

    print("[2] Finding similar local assets...")
    refs = find_references(enhanced, input_dir)
    print(f"[✓] References: {[r.name for r in refs]}")

    print("[3] Generating image...")
    image = generate_image(enhanced, refs)

    print("[4] Converting image to video...")
    frames = image_to_video(image)

    print("[5] Saving result...")
    video_path = save_video(frames, output_dir)
    return video_path

# Entry
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Offline AI Visual Generation Pipeline")
    parser.add_argument("prompt", type=str, help="Prompt for generation")
    parser.add_argument("/input", type=Path, required=True, help="Path to input asset directory")
    parser.add_argument("/output", type=Path, required=True, help="Path to output directory")
    args = parser.parse_args()

    run_pipeline(args.prompt, args.input, args.output)
