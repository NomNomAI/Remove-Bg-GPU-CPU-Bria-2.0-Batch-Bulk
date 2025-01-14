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


Original BRIA-RMBG-2.0 README CONDENSED 
---
license: other
license_name: bria-rmbg-2.0
license_link: https://bria.ai/bria-huggingface-model-license-agreement/
pipeline_tag: image-segmentation
tags:
- remove background
- background
- background-removal
- Pytorch
- vision
- legal liability
- transformers
---

# BRIA Background Removal v2.0 Model Card

RMBG v2.0 is our new state-of-the-art background removal model significantly improves RMBG v1.4. The model is designed to effectively separate foreground from background in a range of
categories and image types. This model has been trained on a carefully selected dataset, which includes:
general stock images, e-commerce, gaming, and advertising content, making it suitable for commercial use cases powering enterprise content creation at scale. 
The accuracy, efficiency, and versatility currently rival leading source-available models. 
It is ideal where content safety, legally licensed datasets, and bias mitigation are paramount. 

Developed by BRIA AI, RMBG v2.0 is available as a source-available model for non-commercial use.

**Purchase:** to purchase a commercial license simply click [Here](https://go.bria.ai/3D5EGp0).

[CLICK HERE FOR A DEMO](https://huggingface.co/spaces/briaai/BRIA-RMBG-2.0)

Join our [Discord community](https://discord.gg/Nxe9YW9zHS) for more information, tutorials, tools, and to connect with other users!



## Model Details
#####
### Model Description

- **Developed by:** [BRIA AI](https://bria.ai/)
- **Model type:** Background Removal 
- **License:** [bria-rmbg-2.0](https://bria.ai/bria-huggingface-model-license-agreement/)
  - The model is released under a Creative Commons license for non-commercial use.
  - Commercial use is subject to a commercial agreement with BRIA.

  **Purchase:** to purchase a commercial license simply click [Here](https://go.bria.ai/3D5EGp0).

- **Model Description:** BRIA RMBG-2.0 is a dichotomous image segmentation model trained exclusively on a professional-grade dataset.
- **BRIA:** Resources for more information: [BRIA AI](https://bria.ai/)



## Training data
Bria-RMBG model was trained with over 15,000 high-quality, high-resolution, manually labeled (pixel-wise accuracy), fully licensed images.
Our benchmark included balanced gender, balanced ethnicity, and people with different types of disabilities.
For clarity, we provide our data distribution according to different categories, demonstrating our model’s versatility.


### Architecture
RMBG-2.0 is developed on the [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) architecture enhanced with our proprietary dataset and training scheme. This training data significantly improves the model’s accuracy and effectiveness for background-removal task.<br>
If you use this model in your research, please cite:

```
@article{BiRefNet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```

