import os
import warnings
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import gradio as gr

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

# Load model (we'll move it to CPU/GPU as needed in the functions)
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
model.eval()

image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

def process_image(input_path, chosen_device):
    try:
        # Move model to chosen device
        model.to(chosen_device)

        # Open/convert the image
        image = Image.open(input_path).convert("RGB")
        original_size = image.size

        # Transform and run inference
        input_tensor = transform_image(image).unsqueeze(0).to(chosen_device)
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()

        # Create mask and apply alpha
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(original_size)
        image.putalpha(mask)

        # Save
        output_path = os.path.join(output_dir, f"no_bg_{os.path.basename(input_path)}")
        image.save(output_path)
        return input_path, output_path
    except Exception as e:
        return input_path, f"Error processing: {str(e)}"

def single_file_process(image, device_choice):
    # Decide CPU or GPU
    if device_choice == "GPU" and torch.cuda.is_available():
        chosen_device = torch.device("cuda")
    else:
        chosen_device = torch.device("cpu")

    input_path, output_path = process_image(image, chosen_device)
    return image, output_path

def batch_process(images, device_choice):
    # Decide CPU or GPU
    if device_choice == "GPU" and torch.cuda.is_available():
        chosen_device = torch.device("cuda")
    else:
        chosen_device = torch.device("cpu")

    results = []
    for image in images:
        input_path, output_path = process_image(image.name, chosen_device)
        results.append((input_path, output_path))
    return results

with gr.Blocks() as demo:
    gr.Markdown("## Remove Bg GPU/CPU FAST Bria 2.0 Batch/Bulk")
    gr.Markdown("This space uses [BRIA Background Removal v2.0](https://huggingface.co/briaai/RMBG-2.0).")

    with gr.Row():
        # Add a device choice radio
        device_choice = gr.Radio(["CPU", "GPU"], value="CPU", label="Select Device")

    with gr.Tab("Single File Processing"):
        with gr.Row():
            input_image = gr.Image(type="filepath", label="Upload Image")
            output_image = gr.Image(label="Processed Image")
        process_button = gr.Button("Process")
        process_button.click(
            single_file_process, 
            inputs=[input_image, device_choice], 
            outputs=[input_image, output_image]
        )

    with gr.Tab("Batch Processing"):
        # Set a fixed height so it becomes scrollable if the list is long
        batch_file_input = gr.File(label="Upload Images", file_types=['image'], file_count="multiple", height=200)
        with gr.Row():
            batch_input_image = gr.Image(label="Current Image")
            batch_output_image = gr.Image(label="Processed Image")
        process_batch_button = gr.Button("Process Batch")

        def batch_process_handler(images, dev_choice):
            results = batch_process(images, dev_choice)
            for input_path, output_path in results:
                # yield results one by one
                yield input_path, output_path

        process_batch_button.click(
            batch_process_handler,
            inputs=[batch_file_input, device_choice],
            outputs=[batch_input_image, batch_output_image]
        ).then(
            fn=lambda: gr.update(value=None),
            inputs=None,
            outputs=batch_file_input
        )

    gr.Markdown(f"Output images are saved in the folder: `{output_dir}`")

demo.launch()
