from pathlib import Path

import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


CHECKPOINT = Path("checkpoints/sam3.pt")
IMAGE_PATH = Path("test_data/person.jpg")
OUTPUT_PATH = Path("test_data/person_mask.png")


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. SAM3 test requires a CUDA GPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading SAM3 from: {CHECKPOINT}")

    model = build_sam3_image_model(checkpoint_path=str(CHECKPOINT), device="cuda")
    processor = Sam3Processor(model)

    image = Image.open(IMAGE_PATH).convert("RGB")
    state = processor.set_image(image)

    print('Running SAM3 prompt: "person"')
    output = processor.set_text_prompt(state=state, prompt="person")

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print(f"Detected objects: {len(masks)}")

    if len(masks) == 0:
        print("No person detected.")
        return

    for index, score in enumerate(scores):
        print(f"Person {index}: score={float(score):.4f}, box={boxes[index].tolist()}")

    # Combine all detected person masks into one binary mask.
    combined_mask = masks.any(dim=0)
    combined_mask = combined_mask.squeeze().detach().cpu().numpy()

    mask_image = Image.fromarray((combined_mask * 255).astype("uint8"), mode="L")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(OUTPUT_PATH)

    print(f"Mask saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
