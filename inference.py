import argparse
import os
from PIL import Image

from milkanything.utils.inference_utils import MilkAnythingInference

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

def process_images(args):
    inferences = MilkAnythingInference(
        checkpoint_path=args.checkpoint,
        direction=args.direction,
        image_size=args.image_size,
        device=args.device
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input):
        img_paths = [args.input]
    elif os.path.isdir(args.input):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        img_paths = [
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {args.input}")

    print(f"\n--- Starting Inference on {len(img_paths)} image(s) ---")

    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)

        try:
            orig, fake = inferences.predict(path)

            if args.compare:
                comparison = Image.new('RGB', (orig.width * 2, orig.height))
                comparison.paste(orig, (0, 0))
                comparison.paste(fake, (orig.width, 0))
                save_path = os.path.join(args.output_dir, f"{name}_compare{ext}")
                comparison.save(save_path)
            else:
                save_path = os.path.join(args.output_dir, f"{name}_translated{ext}")
                fake.save(save_path)

            print(f"[{idx + 1}/{len(img_paths)}] Saved: {save_path}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CycleGAN Inference Script")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the best_cyclegan.pth')
    parser.add_argument('--input', type=str, required=True, help='Path to a single image or an input directory')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Directory to save results')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'], help='Translation direction')
    parser.add_argument('--image_size', type=int, default=256, help='Model input resolution')
    parser.add_argument('--compare', action='store_true', help='Save side-by-side comparison images')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., cuda, cpu)')

    args = parser.parse_args()
    process_images(args)