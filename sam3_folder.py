from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


CHECKPOINT = Path("checkpoints/sam3.pt")
BPE_PATH = Path("sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 person segmentation on every image in a folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for masks. Defaults to <input-dir>/masks.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="person",
        help='SAM3 text prompt (default: "person").',
    )
    return parser.parse_args()


def get_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "masks"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not BPE_PATH.exists():
        raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {BPE_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. SAM3 requires a CUDA GPU.")

    images = get_images(input_dir)
    if not images:
        print(f"No supported images found in: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading SAM3 from: {CHECKPOINT}")
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Images: {len(images)}")
    print(f'Prompt: "{args.prompt}"')

    model = build_sam3_image_model(
        checkpoint_path=str(CHECKPOINT),
        bpe_path=str(BPE_PATH),
        device="cuda",
    )
    processor = Sam3Processor(model)

    for index, image_path in enumerate(images, start=1):
        output_path = output_dir / f"{image_path.stem}.png"

        print(f"[{index}/{len(images)}] {image_path.name}")

        image = Image.open(image_path).convert("RGB")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=args.prompt)

        masks = output["masks"]
        scores = output["scores"]

        if len(masks) == 0:
            print("  No objects detected -> saving all-white mask")
            mask = torch.zeros(
                (image.height, image.width), dtype=torch.bool, device="cpu"
            )
        else:
            print(f"  Detected objects: {len(masks)}")
            print(
                "  Scores: "
                + ", ".join(f"{float(score):.4f}" for score in scores)
            )
            mask = masks.any(dim=0).squeeze().detach().cpu()

        # Output convention: black = detected object, white = background.
        inverted_mask = (~mask.bool()).numpy().astype("uint8") * 255
        Image.fromarray(inverted_mask, mode="L").save(output_path)
        print(f"  Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
