# import random

# with open("data/raw/clean_hi.txt", encoding="utf-8") as f_hi, \
#      open("data/raw/clean_ms.txt", encoding="utf-8") as f_ms:
#     hi_lines = f_hi.readlines()
#     ms_lines = f_ms.readlines()

# sample_idx = random.sample(range(len(hi_lines)), 30)
# for i in sample_idx:
#     print(f"HI: {hi_lines[i].strip()}")
#     print(f"MS: {ms_lines[i].strip()}")
#     print("---")

##################################################
#tokenize check
# from datasets import load_from_disk
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# train_ds = load_from_disk("data/tokenized/train")

# example = train_ds[0]
# print("Input IDs:", example["input_ids"][:20])
# print("Decoded input:", tokenizer.decode(example["input_ids"], skip_special_tokens=True))

# print("Decoded labels:", tokenizer.decode(example["labels"], skip_special_tokens=True))
#############################################
# datasize check
from datasets import load_from_disk
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path(".")
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"

train_ds = load_from_disk(TOKENIZED_DIR / "train")
val_ds = load_from_disk(TOKENIZED_DIR / "validation")

print("Train size:", len(train_ds))
print("Validation size:", len(val_ds))