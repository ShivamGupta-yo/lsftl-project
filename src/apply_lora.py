from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "facebook/nllb-200-distilled-600M"

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=False)
    return model, tokenizer

def attach_lora(model, r=16, alpha=32, dropout=0.05):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        bias="none",
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model

if __name__ == "__main__":
    model, tokenizer = load_base_model()
    lora_model = attach_lora(model)

    lora_model.print_trainable_parameters()