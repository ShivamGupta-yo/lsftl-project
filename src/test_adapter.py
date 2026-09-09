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
    print("GPU count visible:", torch.cuda.device_count())
    return device

def load_finetuned_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(device)
    return model, tokenizer

def translate(model, tokenizer, device, text, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=50
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    device = get_device()

    model, tokenizer = load_finetuned_model(device)
    print("Fine-tuned LoRA adapter loaded successfully.")

    test_sentences = [
        "नमस्ते, आप कैसे हैं?",
        "मुझे पसंद नहंक ईमुझेधमक दे।",
        "हम एक समस्यहै.",
        "तुमने सगईकर ल?",
    ]

    for sentence in test_sentences:
        result = translate(model, tokenizer, device, sentence)
        print(f"HI: {sentence}")
        print(f"MS: {result}")
        print("---")