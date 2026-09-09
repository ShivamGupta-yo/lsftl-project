import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

MODEL_NAME = "facebook/nllb-200-distilled-600M"
ADAPTER_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "lsftl-hi-ms" / "final"

SRC_LANG = "hin_Deva"
TGT_LANG = "zsm_Latn"

def load_finetuned_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)

    # Load and attach the trained LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

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
    model, tokenizer, device = load_finetuned_model()
    print("Fine-tuned LoRA adapter loaded successfully.")
    print("Using device:", device)

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