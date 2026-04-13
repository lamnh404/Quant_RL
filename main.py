import os
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.quantizer import QwenGPTQQuantizer
from src.grpo_trainer import train_r3_quant_grpo
from src.sft_trainer import train_sft_baseline

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QUANT_BITS = 3
def setup_environment():
    print("--- 1. Khởi tạo cấu trúc thư mục ---")
    directories = ["data/science_qa", "weights", "r3_quant_checkpoints", "sft_baseline_checkpoints"]
    for folder in directories:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Đã tạo thư mục: {folder}")

def download_data():
    print("\n--- 2. Đang tải/đọc Dataset ScienceQA ---")
    target_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    if not os.path.exists(target_path):
        print("Đang tải dataset từ Hugging Face...")
        dataset = load_dataset("derek-thomas/ScienceQA", split="validation")
        dataset.to_parquet(target_path)
        print(f"Đã lưu dataset tại: {target_path}")
    else:
        print(f"Dataset đã tồn tại tại {target_path}, đang load...")
        dataset = load_dataset("parquet", data_files=target_path, split="train")
    return dataset

def download_sft_data():
    print("\n--- 2.5 Đang tải Dataset Mini CoT 8k (Cho pha SFT) ---")
    print("Đang tải dataset từ Hugging Face...")
    sft_dataset = load_dataset("luodian/mini_cot_8k_verified", split="train")
    return sft_dataset

def download_model(model_id):
    print(f"\n--- 3. Đang tải Model {model_id} ---")
    model_name = model_id.split("/")[-1]
    local_dir = f"./weights/{model_name}"
    
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"Đang tải model {model_id} (quá trình này có thể lâu)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main"
        )
        print(f"Model đã được tải về: {local_dir}")
    else:
        print(f"Model đã tồn tại ở: {local_dir}")
    return local_dir

def run_quantization(base_model_dir, dataset_path, bits):
    model_name = os.path.basename(base_model_dir)
    save_dir = f"./weights/{model_name}-GPTQ-Int{bits}"
    
    print(f"\n--- 4. Bắt đầu lượng tử hoá model (GPTQ Int{bits}) ---")
    if not os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"Đang lượng tử hóa và lưu tại: {save_dir}")
        quantizer = QwenGPTQQuantizer(base_model_dir, save_dir, dataset_path)
        quantizer.quantize_and_save(bits=bits)
        print("[SUCCESS] Quá trình lượng tử hóa hoàn tất thành công!")
    else:
        print(f"Model lượng tử hóa đã tồn tại ở: {save_dir}")
        
    return save_dir

def run_rl_training(quant_model_dir, sft_dataset, grpo_dataset):
    sft_output_dir = "./sft_baseline_checkpoints"
    grpo_output_dir = "./r3_quant_checkpoints"
    
    checkpoint_exists = os.path.exists(sft_output_dir) and \
                        os.path.exists(os.path.join(sft_output_dir, "adapter_config.json"))

    if checkpoint_exists:
        print(f"\n--- [SKIP] Đã tìm thấy SFT Checkpoint tại {sft_output_dir}. Bỏ qua bước train SFT. ---")
    else:
        print("\n--- 5. Bắt đầu quá trình Supervised Fine-Tuning (SFT) với Mini CoT ---")
        train_sft_baseline(quant_model_dir, sft_dataset, sft_output_dir)
        print(f"\n[SUCCESS] Hoàn tất quá trình huấn luyện SFT! Model được lưu tại: {sft_output_dir}")
    
    print("\n--- 6. Bắt đầu quá trình huấn luyện RL (GRPO) tiếp nối bước SFT ---")
    train_r3_quant_grpo(quant_model_dir, grpo_dataset, grpo_output_dir, sft_lora_dir=sft_output_dir)
    print(f"\n[SUCCESS] Hoàn tất quá trình huấn luyện GRPO! Model được lưu tại: {grpo_output_dir}")

def main():
    print("==========================================")
    print(" BẮT ĐẦU PIPELINE END-TO-END QUANT & RL")
    print(f" MODEL: {BASE_MODEL_ID}")
    print(" PIPELINE: QUANTIZE -> SFT -> GRPO")
    print("==========================================\n")
    
    setup_environment()
    grpo_dataset = download_data()
    sft_dataset = download_sft_data()
    
    dataset_path = "./data/science_qa/validation-00000-of-00001-6c7328ff6c84284c.parquet"
    
    base_model_dir = download_model(BASE_MODEL_ID)
    quant_model_dir = run_quantization(base_model_dir, dataset_path, QUANT_BITS)
    
    run_rl_training(quant_model_dir, sft_dataset, grpo_dataset)

if __name__ == "__main__":
    main()
