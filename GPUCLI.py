import os
import warnings
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import shutil

# Suppress warnings
warnings.filterwarnings('ignore')

# Print PyTorch and CUDA versions for debugging
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print(f"PyTorch version: {TORCH_VERSION}, CUDA version: {CUDA_VERSION}")

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable optimizations where available
torch.set_float32_matmul_precision('high')

# Load the model
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
model.to(device)
model.eval()

# Data settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Directory handling
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "input")
output_dir = os.path.join(script_dir, "output")
done_dir = os.path.join(script_dir, "done")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(done_dir, exist_ok=True)

def process_image(input_path, output_path):
    try:
        # Load and convert image
        image = Image.open(input_path).convert("RGB")
        original_size = image.size

        # Transform and process
        input_tensor = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()

        # Convert prediction to mask
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(original_size)

        # Apply mask and save
        image.putalpha(mask)
        image.save(output_path)
        print(f"Processed: {os.path.basename(output_path)}")

    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False
    return True

def main():
    # Process images from the input folder
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the input directory")
        return

    total = len(image_files)
    successful = 0

    print(f"Found {total} images to process")

    for file_name in image_files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"no_bg_{file_name}")

        if process_image(input_path, output_path):
            successful += 1
            # Move processed image to done folder
            done_path = os.path.join(done_dir, file_name)
            shutil.move(input_path, done_path)

    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful}/{total} images")
    print(f"Output saved to: {output_dir}")
    print(f"Processed images moved to: {done_dir}")

if __name__ == "__main__":
    main()
