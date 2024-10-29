# Using-Stable-Diffusion-in-Google-Colab

This repository provides a guide for generating images using Stable Diffusion within Google Colab. It leverages the Hugging Face `diffusers` library and supports customization through LoRA.


## Introduction

This README explains how to generate images using Stable Diffusion in Google Colab.  It provides a step-by-step guide, from setup to image generation and LoRA implementation.


## Setup

### 1. Mount Google Drive

Mount your Google Drive to save generated images and load training data.

```python
from google.colab import drive
drive.mount('/content/drive')

# Set your working directory (adjust the path as needed)
%cd /content/drive/MyDrive/Colab Notebooks/【画像生成】Stable Dffusion

# Create an image storage folder
import os
os.makedirs("/content/drive/MyDrive/Colab Notebooks/【画像生成】Stable Dffusion/Image_storage_folder", exist_ok=True)
```

### 2. Install Libraries

Install the required libraries.

```bash
!pip install diffusers transformers peft accelerate bitsandbytes
```

### 3. Log in to Hugging Face

Log in to Hugging Face to access the models.

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Running Stable Diffusion

### Choosing a Pipeline

Select one of the following pipelines:

#### 1. DiffusionPipeline (General)

```python
from diffusers import DiffusionPipeline
import torch

model_id = 'gsdf/Counterfeit-V2.5' # Specify the model ID
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
```

#### 2. StableDiffusionPipeline (Stable Diffusion Specific)

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "stabilityai/stable-diffusion-2"  # Specify the model ID
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
```

**`torch_dtype=torch.float16` and `.to("cuda")`**: These options utilize your GPU for faster generation and reduce memory usage. If you encounter issues, try removing them.  If your Colab instance doesn't have a GPU, remove `.to("cuda")`.


### Generating Images

#### Defining Prompts

```python
# Define your positive prompt
POSITIVE_PROMPT = "Enter your prompt here"

# Define your negative prompt (pre-filled with defaults, customize as needed)
NEGATIVE_PROMPT = "low quality,worst quality,out of focus,ugly,error,JPEG artifacts,low resolution,blurry,bokeh," \
                  "pubic hair,bad anatomy,long_neck,long_body,longbody,deformed,mutated,disfigured,missing arms," \
                  "extra_arms,mutated hands,extra_legs,bad hands,poorly_drawn_hands,malformed_hands," \
                  "missing_limb,floating_limbs,disconnected limbs,extra_fingers,bad fingers,liquid fingers," \
                  "missing fingers,extra digit,fewer digits,ugly face,deformed eyes," \
                  "partial face,partial head,bad face,inaccurate limbs,cropped,wrong perspective,bad proportions," \
                  "oversized limbs,undersized limbs,distorted limbs,twisted body,asymmetrical face," \
                  "misaligned features,weird expressions,uncanny smile,closed eyes when they should be open," \
                  "extra joints,missing joints,disconnected body parts,text,signature,watermark," \
                  "username,artist name,stamp,title,subtitle,date," \
                  "open mouth,half-open eyes,oil painting,sketch,watercolor,2D,flat color,plastic look," \
                  "doll-like,waxy texture,cartoonish,uncanny valley," \
                  "unnatural skin texture,artificial lighting,unrealistic shadows,uncanny skin tone," \
                  "unnatural pose,stiff posture,artificial background,synthetic looking,artificial looking"

# Helper function to handle long prompts (defined in the notebook)
positive_embeds, negative_embeds = token_auto_concat_embeds(pipe, POSITIVE_PROMPT, NEGATIVE_PROMPT)

```

#### Running Image Generation

```python
image = pipe(prompt_embeds=positive_embeds,
             negative_prompt_embeds=negative_embeds,
             height=768, width=512, guidance_scale=7.5,
             num_images_per_prompt=1).images[0]
```

Control the generated image by adjusting parameters like `guidance_scale`, `height`, `width`, and `num_images_per_prompt`.  

See the [official documentation](https://huggingface.co/docs/diffusers/main/en/index) for more details.



#### Saving the Image

```python
image.save("/content/drive/MyDrive/Colab Notebooks/【画像生成】Stable Dffusion/Image_storage_folder/generated_image.png")
```


## Applying LoRA (Optional)

### Loading a Pre-trained LoRA

```python
from peft import PeftModel, LoraConfig, get_peft_model

# Configure LoRA ...

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.unet = get_peft_model(pipe.unet, lora_config)

pipe.unet.load_state_dict(torch.load("/path/to/your/lora_weights.bin"), strict=False)  # Specify the path to your LoRA weights
pipe.to("cuda")
```

Refer to the `3(番外編).LoRAを導入してDiffusionPipelineを使用する場合` section in the notebook for instructions on training your own LoRA.


## License

This repository is licensed under the MIT License.
