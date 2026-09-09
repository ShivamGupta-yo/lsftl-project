import os
# Pin strictly to GPU index 3 before loading any CUDA-dependent libraries
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from apply_lora import load_base_model, attach_lora

MODEL_NAME = "facebook/nllb-200-distilled-600M"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "lsftl-hi-ms"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model, _ = load_base_model()
    lora_model = attach_lora(model)
    lora_model.print_trainable_parameters()

    train_ds = load_from_disk(TOKENIZED_DIR / "train")
    val_ds = load_from_disk(TOKENIZED_DIR / "validation")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=lora_model,
        padding=True
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        predict_with_generate=True,
        report_to="none",
        local_rank=-1,
    )

    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    final_checkpoint = CHECKPOINT_DIR / "final"
    lora_model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)

    print(f"Training complete. Adapter saved to {final_checkpoint}")

if __name__ == "__main__":
    main()