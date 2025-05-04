import os

import torch
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from huggingface_hub import HfFolder
from loguru import logger

# --------------------------------------------------------------------------
# Đặt token của bạn vào đây
hf_token = os.getenv("HF_TOKEN", "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")
# Lưu token vào local
HfFolder.save_token(hf_token)

from huggingface_hub import login

hf_access_token = "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"
login(token=hf_access_token)


def _load():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        print("CUDA is available.")
        model_id = "stabilityai/stable-diffusion-3.5-medium"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=dtype,
            device_map="auto",
        )


_load()
logger.info("Model loaded.")
