![{7F6DBCC6-D6C6-4725-B436-6D200287B1FC}](https://github.com/user-attachments/assets/67706cf7-a307-4a33-96a8-dc139172d760)


Remove Bg GPU/CPU FAST Bria 2.0 Batch/Bulk
Short intro: This tool removes image backgrounds in bulk using BRIA’s RMBG v2.0.
License: BRIA RMBG-2.0 is source-available for non-commercial use. Commercial use requires a license from BRIA.
Credits: Model by BRIA AI (https://briaglobal.com) and references to the BiRefNet paper.


Simple lightweight fast Background removal. No fancy bells and whistles.

Three versions included. 
CPU CLI
GPU CLI
Gradio GUI

Gradio allows choosing of files. CLI will use the input folder

Create and activate a Python virtual environment:

python -m venv venv
venv\Scripts\activate

Install packages:

pip install --upgrade pip
pip install -r requirements.txt

GPU support:
install PyTorch/Torchvision with CUDA support compatible with your hardware, e.g. CUDA 11.8:

pip install --index-url https://download.pytorch.org/whl/cu118 --upgrade torch torchvision torchaudio

Run the script:

python gradiover.py




