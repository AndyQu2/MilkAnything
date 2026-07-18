import os

# Suppress non-essential bitsandbytes library validation warnings during inference
os.environ["BNB_CUDA_VERSION"] = "0"
os.environ["FORCE_CONTAINERS"] = "1"

import argparse
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import OpenposeDetector, CannyDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Nai-Long LoRA Inference with Dynamic Prompt Layering (2026)")

    # User-defined text description to overlay on top of the embedded template
    parser.add_argument("--user_prompt", type=str, default="",
                        help="Additional text prompt layer to overlay on top of the embedded template (e.g., 'holding an apple', 'typing on a laptop')")

    # Generic base parameters
    parser.add_argument("--lora_path", type=str, default="./lora_weight",
                        help="Path to the LoRA weights directory or single safetensors file")
    parser.add_argument("--lora_scale", type=float, default=0.8, help="LoRA conditioning scale (0.0 - 1.0)")
    parser.add_argument("--output", type=str, default="output.png", help="Filename of the output image")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-Free Guidance (CFG) scale")

    # ControlNet specific parameters
    parser.add_argument("--use_controlnet", action="store_true",
                        help="Enable dual ControlNet (Pose + Background Canny)")
    parser.add_argument("--ref_image", type=str, default="human.jpg", help="Path to the reference human pose image")
    parser.add_argument("--pose_scale", type=float, default=0.75, help="Weight for OpenPose control")
    parser.add_argument("--canny_scale", type=float, default=0.55, help="Weight for Background Canny control")

    return parser.parse_args()


def main():
    args = parse_args()

    # Adaptive hardware acceleration selection compatible with 2026 platforms
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # Keep float32 for Apple Silicon MPS stability
    else:
        device = "cpu"
        torch_dtype = torch.float32

    # ==========================================
    # 🧩 2026 Prompt Layering Mechanism
    # ==========================================
    # Base core template to lock model subject identity and visual styling constraints
    base_template = (
        "A high-quality macro photograph of a cute yellow chubby dinosaur-like dragon toy, "
        "big round glossy green eyes, oversized white belly, soft smooth plastic texture, "
        "standing in a grey modern office room with windows, realistic shadows, highly detailed, 3d vinyl toy trend"
    )

    negative_prompt = "wings, horns, evil, aggressive, dragon scales, spikes, scary, sharp teeth, photo, deformed"

    # Weave user custom description as a structural text layer on top of base configurations
    if args.user_prompt.strip():
        # Optimization block: Appending specific interaction actions directly after core subject definition
        user_description = args.user_prompt.strip().rstrip(",")
        final_prompt = f"{base_template}, {user_description}"
    else:
        final_prompt = base_template

    print("==================================================")
    print(f"🧬 Final Layered Prompt: \n\"{final_prompt}\"")
    print("==================================================")

    # ==========================================
    # MODE 1: Modern Multi-ControlNet Fusion
    # ==========================================
    if args.use_controlnet:
        print(f"🎬 Mode activated: [Dual ControlNet Fusion Mode]")
        if not os.path.exists(args.ref_image):
            raise FileNotFoundError(f"Reference image not found at: {args.ref_image}")

        user_image = load_image(args.ref_image).resize((512, 512))

        print("🩻 Extracting human pose skeleton...")
        pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        pose_skeleton = pose_detector(user_image, include_hand=True, include_face=True)

        print("📐 Extracting background edge map (Canny)...")
        canny_detector = CannyDetector()
        bg_canny = canny_detector(user_image, low_threshold=100, high_threshold=200)

        print("📦 Loading ControlNet pretrained weights...")
        controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch_dtype)
        controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch_dtype)

        print("🚀 Initializing ControlNet Pipeline...")
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=[controlnet_pose, controlnet_canny],
            torch_dtype=torch_dtype
        ).to(device)

    # ==========================================
    # MODE 2: Pure LoRA Text-to-Image
    # ==========================================
    else:
        print(f"📝 Mode activated: [Pure LoRA Text-to-Image Mode]")
        print("🚀 Initializing standard StableDiffusion Pipeline...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype
        ).to(device)

    # ==========================================
    # ⚙️ Unified Pipeline Configuration & LoRA Injection
    # ==========================================
    pipeline.safety_checker = None
    # Enable attention slicing for optimized VRAM consumption
    pipeline.enable_attention_slicing()

    print(f"💉 Injecting LoRA adapter weights from: {args.lora_path} ...")
    # Modern standard PEFT weight management API calls
    pipeline.load_lora_weights(args.lora_path, adapter_name="nailong")
    pipeline.set_adapters(["nailong"], adapter_weights=[args.lora_scale])

    # Construct the base parameter settings map for generating models
    generation_kwargs = {
        "prompt": final_prompt,  # Pass the combined dual-layered prompt configuration
        "negative_prompt": negative_prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": 512,
        "width": 512,
    }

    if args.use_controlnet:
        generation_kwargs.update({
            "image": [pose_skeleton, bg_canny],
            "controlnet_conditioning_scale": [args.pose_scale, args.canny_scale]
        })

    # Execute high-efficiency forward graph generation pass
    print("🎨 Rendering image with model integration...")
    with torch.inference_mode():
        image = pipeline(**generation_kwargs).images[0]

    # Output file preservation pass
    image.save(args.output)
    print(f"✨ Generation complete! Image saved to: {args.output}")


if __name__ == "__main__":
    main()