# Using-Stable-Diffusion-in-Google-Colab

このリポジトリは、Google Colab上でStable Diffusionを用いて画像を生成するための手順をまとめたものです。

Hugging Faceの`diffusers`ライブラリを使用し、LoRAによるカスタマイズにも対応しています。


## はじめに

このREADMEでは、Google ColabでStable Diffusionを使って画像を生成する方法を説明します。

事前準備から画像生成、LoRAの適用まで、ステップバイステップで解説します。


## 事前準備

### 1. Google Driveのマウント

生成した画像を保存したり、学習データを読み込んだりする為に、Google Driveをマウントします。

```python
from google.colab import drive
drive.mount('/content/drive')

# 作業ディレクトリの設定 (ご自身の環境に合わせてパスを変更してください)
%cd /content/drive/MyDrive/Colab Notebooks/【画像生成】Stable Dffusion

# 画像保存フォルダの作成
import os
os.makedirs("/content/drive/MyDrive/Colab Notebooks/【画像生成】Stable Dffusion/Image_storage_folder", exist_ok=True)
```

### 2. ライブラリのインストール

必要なライブラリをインストールします。

```bash
!pip install diffusers transformers peft accelerate bitsandbytes
```

### 3. Hugging Faceへのログイン

Hugging Faceのモデルを利用するためにログインします。

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Stable Diffusionの実行

### パイプラインの選択

以下のいずれかのパイプラインを選択して使用します。

#### 1. DiffusionPipeline (汎用)

```python
from diffusers import DiffusionPipeline
import torch

model_id = 'gsdf/Counterfeit-V2.5' # モデルIDを指定
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
```

#### 2. StableDiffusionPipeline (Stable Diffusion専用)

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "stabilityai/stable-diffusion-2"  # モデルIDを指定
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
```

**`torch_dtype=torch.float16` and `.to("cuda")`**:  These options utilize your GPU for faster generation and reduce memory usage.  If you encounter issues, try removing them.


### 画像生成

#### プロンプトの定義

```python
# ポジティブプロンプトを定義
POSITIVE_PROMPT = "ここにプロンプトを入力"

# ネガティブプロンプトを定義
# デフォルトで入力済み（生成したい画像に合わせて変更してください）
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

# 長いプロンプトを扱うためのヘルパー関数 (notebook内に定義されています)
positive_embeds, negative_embeds = token_auto_concat_embeds(pipe, POSITIVE_PROMPT, NEGATIVE_PROMPT)
```

#### 画像生成の実行

```python
image = pipe(prompt_embeds=positive_embeds,
             negative_prompt_embeds=negative_embeds,
             height=768, width=512, guidance_scale=7.5,
             num_images_per_prompt=1).images[0]
```

`guidance_scale`、`height`、`width`、`num_images_per_prompt` 等のパラメータを調整することで、生成画像を制御できます。 

詳しくは[公式ドキュメント](https://huggingface.co/docs/diffusers/main/en/index)を参照ください。


#### 画像の保存

```python
image.save("/content/drive/MyDrive/Colab Notebooks/【画像生成】Stable Dffusion/Image_storage_folder/生成画像.png")
```


## LoRAの適用 (オプション)

### 学習済みLoRAのロード

```python
from peft import PeftModel, LoraConfig, get_peft_model

# LoRAの設定 ...

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.unet = get_peft_model(pipe.unet, lora_config)

pipe.unet.load_state_dict(torch.load("/path/to/your/lora_weights.bin"), strict=False)  # LoRAの重みのパスを指定
pipe.to("cuda")
```

LoRAの学習方法については、notebook内の`3(番外編).LoRAを導入してDiffusionPipelineを使用する場合`セクションを参照ください。


## ライセンス

このリポジトリはMITライセンスで公開されています。
