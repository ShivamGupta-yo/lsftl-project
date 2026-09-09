import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model_and_tokenizer(model_name="facebook/nllb-200-distilled-600M"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

def test_translation(
    model,
    tokenizer,
    device,
    text,
    src_lang="hin_Deva",
    tgt_lang="zsm_Latn"
):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id,
        max_length=50
    )
    return tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )

if __name__ == "__main__":
    model, tokenizer, device = load_model_and_tokenizer()
    print("Model and tokenizer loaded successfully.")
    print("Using device:", device)
    result = test_translation(
        model,
        tokenizer,
        device,
        "नमस्ते, आप कैसे हैं?"
    )
    print("Translation output:", result)