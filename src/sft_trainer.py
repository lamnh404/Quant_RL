import torch
import sys
import os
from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, TrainerCallback
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.utils import prepare_minicap_for_sft

class SFTProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"📊 [SFT Progress] Step {state.global_step}/{state.max_steps} | Loss: {logs['loss']:.4f} | LR: {logs.get('learning_rate', 0):.2e}")

def train_sft_baseline(model_dir: str, train_data, output_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 512 * 28 * 28
    peft_model = apply_lora_to_quantized_model(model_dir)
    
    sft_dataset = prepare_minicap_for_sft(train_data)

    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="messages", 
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
        learning_rate=2e-5,          
        lr_scheduler_type="cosine",
        logging_steps=1,           
        max_steps=500, # adjust this
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        gradient_checkpointing=True, 
        bf16=True,                   
        remove_unused_columns=False, 
        report_to="none",
 
    )

    trainer = SFTTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=sft_dataset,
        callbacks=[SFTProgressCallback()],
    )

    trainer.train()
    
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir) 

