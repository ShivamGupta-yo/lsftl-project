from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "facebook/nllb-200-distilled-600M"

SRC_LANG = "hin_Deva"   # Hindi
TGT_LANG = "zsm_Latn"   # Malay

MAX_LENGTH = 128
RANDOM_STATE = 42


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# Project root = lsftl-project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"

HI_FILE = DATA_RAW_DIR / "clean_hi.txt"
MS_FILE = DATA_RAW_DIR / "clean_ms.txt"


# --------------------------------------------------
# Load cleaned parallel data
# --------------------------------------------------

def load_cleaned_pairs(hi_path, ms_path):

    with open(hi_path, encoding="utf-8") as f_hi, \
         open(ms_path, encoding="utf-8") as f_ms:

        hi_lines = [line.strip() for line in f_hi.readlines()]
        ms_lines = [line.strip() for line in f_ms.readlines()]

    if len(hi_lines) != len(ms_lines):
        raise ValueError(
            f"Number of Hindi and Malay lines does not match: "
            f"Hindi={len(hi_lines)}, Malay={len(ms_lines)}"
        )

    return hi_lines, ms_lines


# --------------------------------------------------
# Build train / validation / test datasets
# --------------------------------------------------

def build_dataset_dict(hi_lines, ms_lines):

    # 80% train, 10% validation, 10% test

    hi_train, hi_temp, ms_train, ms_temp = train_test_split(
        hi_lines,
        ms_lines,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    hi_val, hi_test, ms_val, ms_test = train_test_split(
        hi_temp,
        ms_temp,
        test_size=0.5,
        random_state=RANDOM_STATE
    )

    return {
        "train": Dataset.from_dict({
            "source": hi_train,
            "target": ms_train
        }),

        "validation": Dataset.from_dict({
            "source": hi_val,
            "target": ms_val
        }),

        "test": Dataset.from_dict({
            "source": hi_test,
            "target": ms_test
        }),
    }


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize_function(examples, tokenizer):

    # -----------------------------
    # Tokenize Hindi source
    # -----------------------------

    tokenizer.src_lang = SRC_LANG

    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # -----------------------------
    # Tokenize Malay target
    # -----------------------------

    tokenizer.src_lang = TGT_LANG

    labels = tokenizer(
        examples["target"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # -----------------------------
    # Ignore padding in loss
    # -----------------------------

    labels["input_ids"] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Project root:", PROJECT_ROOT)
    print("Hindi file:", HI_FILE)
    print("Malay file:", MS_FILE)

    # Check that input files exist
    if not HI_FILE.exists():
        raise FileNotFoundError(f"Hindi file not found: {HI_FILE}")

    if not MS_FILE.exists():
        raise FileNotFoundError(f"Malay file not found: {MS_FILE}")

    # -----------------------------
    # Load tokenizer
    # -----------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=SRC_LANG
    )

    print("Tokenizer loaded successfully.")

    # -----------------------------
    # Load cleaned sentence pairs
    # -----------------------------

    hi_lines, ms_lines = load_cleaned_pairs(
        HI_FILE,
        MS_FILE
    )

    print(f"\nLoaded {len(hi_lines)} parallel sentence pairs.")

    # -----------------------------
    # Train / validation / test split
    # -----------------------------

    dataset_dict = build_dataset_dict(
        hi_lines,
        ms_lines
    )

    print("\nDataset split:")
    print("Train size:", len(dataset_dict["train"]))
    print("Validation size:", len(dataset_dict["validation"]))
    print("Test size:", len(dataset_dict["test"]))

    # -----------------------------
    # Tokenize each split
    # -----------------------------

    tokenized_datasets = {}

    print("\nTokenizing datasets...")

    for split, ds in dataset_dict.items():

        print(f"Tokenizing {split}...")

        tokenized_datasets[split] = ds.map(
            lambda examples: tokenize_function(
                examples,
                tokenizer
            ),
            batched=True,
            remove_columns=["source", "target"]
        )

    # -----------------------------
    # Create output directories
    # -----------------------------

    TOKENIZED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------
    # Save tokenized datasets
    # -----------------------------

    for split, ds in tokenized_datasets.items():

        output_path = TOKENIZED_DIR / split

        ds.save_to_disk(output_path)

        print(
            f"Saved {split} dataset to: {output_path}"
        )

    print("\nTokenization completed successfully.")