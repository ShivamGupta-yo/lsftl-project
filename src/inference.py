import os
# Pin strictly to GPU index 3 before loading any CUDA-dependent libraries
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-600M"
ADAPTER_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "lsftl-hi-ms" / "final"

SRC_LANG = "hin_Deva"
TGT_LANG = "zsm_Latn"

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run on CPU.")
    device = torch.device("cuda")
    print("Using device:", device)
    print("GPU name:", torch.cuda.get_device_name(device))
    return device

def load_base_and_lora_model(device):
    print("Loading base model and LoRA adapter, please wait...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load base model twice: once to stay as pure baseline, once to attach LoRA onto
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)
    base_model = base_model.to(device)
    base_model.eval()

    lora_base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)
    lora_model = PeftModel.from_pretrained(lora_base, ADAPTER_PATH)
    lora_model = lora_model.to(device)
    lora_model.eval()

    return base_model, lora_model, tokenizer

def verify_lora_active(lora_model):
    is_peft_model = isinstance(lora_model, PeftModel)
    print(f"Is PEFT/LoRA model: {is_peft_model}")

    if is_peft_model:
        trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in lora_model.parameters())
        print(f"Adapter loaded from: {ADAPTER_PATH}")
        print(f"Trainable (LoRA) params: {trainable_params:,} / {total_params:,}")
    else:
        raise RuntimeError("Model is NOT a PEFT/LoRA model — something went wrong loading the adapter.")

def translate(model, tokenizer, device, text):
    tokenizer.src_lang = SRC_LANG
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    target_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=128
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

if __name__ == "__main__":
    device = get_device()
    base_model, lora_model, tokenizer = load_base_and_lora_model(device)
    verify_lora_active(lora_model)

    print("\n" + "=" * 50)
    print("LSFTL Hindi -> Malay Live Translation Demo")
    print("Shows Baseline vs LSFTL (LoRA fine-tuned) side by side.")
    print("Type a Hindi sentence and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50 + "\n")

    while True:
        text = input("Hindi input: ").strip()

        if text.lower() in ("quit", "exit"):
            print("Exiting demo.")
            break

        if not text:
            continue

        base_translation = translate(base_model, tokenizer, device, text)
        lora_translation = translate(lora_model, tokenizer, device, text)

        print(f"Baseline (no LoRA): {base_translation}")
        print(f"LSFTL (with LoRA):  {lora_translation}\n")