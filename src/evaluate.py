import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import sacrebleu
from comet import download_model, load_from_checkpoint

MODEL_NAME = "facebook/nllb-200-distilled-600M"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "checkpoints" / "lsftl-hi-ms" / "final"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"

SRC_LANG = "hin_Deva"
TGT_LANG = "zsm_Latn"
BATCH_SIZE = 8

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run on CPU.")
    device = torch.device("cuda")
    print("Using device:", device)
    print("GPU name:", torch.cuda.get_device_name(device))
    print("GPU count visible:", torch.cuda.device_count())
    return device

def load_model(device, use_adapter):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)

    if use_adapter:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        model = base_model

    model = model.to(device)
    model.eval()
    return model, tokenizer

def generate_translations(model, tokenizer, device, source_texts):
    tokenizer.src_lang = SRC_LANG
    target_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

    predictions = []
    for i in range(0, len(source_texts), BATCH_SIZE):
        batch = source_texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=128
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        predictions.extend(decoded)
        print(f"Translated {i + len(batch)}/{len(source_texts)}")

    return predictions

def compute_bleu_chrf(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return bleu.score, chrf.score

def compute_comet(sources, predictions, references):
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    output = comet_model.predict(data, batch_size=8, gpus=1)
    return output.system_score

def run_evaluation(label, use_adapter, device, tokenizer, source_texts, reference_texts):
    print(f"\n=== Evaluating: {label} ===")
    model, tokenizer = load_model(device, use_adapter)

    predictions = generate_translations(model, tokenizer, device, source_texts)

    bleu, chrf = compute_bleu_chrf(predictions, reference_texts)
    comet = compute_comet(source_texts, predictions, reference_texts)

    print(f"\n--- Results: {label} ---")
    print(f"BLEU:  {bleu:.2f}")
    print(f"chrF:  {chrf:.2f}")
    print(f"COMET: {comet:.4f}")

    return {"label": label, "bleu": bleu, "chrf": chrf, "comet": comet, "predictions": predictions}

if __name__ == "__main__":
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_ds = load_from_disk(TOKENIZED_DIR / "test")

    # Decode tokenized test set back into raw text for evaluation
    source_texts = [tokenizer.decode(ex, skip_special_tokens=True) for ex in test_ds["input_ids"]]
    reference_texts = [
        tokenizer.decode([t for t in ex if t != -100], skip_special_tokens=True)
        for ex in test_ds["labels"]
    ]

    print(f"Loaded {len(source_texts)} test examples")

    baseline_results = run_evaluation(
        "Baseline (no LoRA)", use_adapter=False,
        device=device, tokenizer=tokenizer,
        source_texts=source_texts, reference_texts=reference_texts
    )

    lsftl_results = run_evaluation(
        "LSFTL (LoRA fine-tuned)", use_adapter=True,
        device=device, tokenizer=tokenizer,
        source_texts=source_texts, reference_texts=reference_texts
    )

    print("\n=== Summary ===")
    print(f"{'Metric':<10}{'Baseline':<15}{'LSFTL':<15}")
    print(f"{'BLEU':<10}{baseline_results['bleu']:<15.2f}{lsftl_results['bleu']:<15.2f}")
    print(f"{'chrF':<10}{baseline_results['chrf']:<15.2f}{lsftl_results['chrf']:<15.2f}")
    print(f"{'COMET':<10}{baseline_results['comet']:<15.4f}{lsftl_results['comet']:<15.4f}")