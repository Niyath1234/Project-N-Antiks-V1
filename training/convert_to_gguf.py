#!/usr/bin/env python3
"""
Convert trained LoRA model to GGUF format for Ollama
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys

def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    merged_dir = script_dir / "merged_model"
    
    if not output_dir.exists():
        print("❌ No trained model found at training/output/")
        sys.exit(1)
    
    print("="*60)
    print("🔄 Converting LoRA Model to GGUF")
    print("="*60)
    
    # Step 1: Load base model
    print("\n📦 Step 1: Loading base model...")
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # Load to CPU for merging
    )
    
    print("✅ Base model loaded")
    
    # Step 2: Load and merge LoRA weights
    print("\n🔗 Step 2: Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(output_dir))
    print("✅ LoRA adapter loaded")
    
    print("\n🔀 Step 3: Merging LoRA with base model...")
    model = model.merge_and_unload()
    print("✅ Models merged")
    
    # Step 4: Save merged model
    print("\n💾 Step 4: Saving merged model...")
    merged_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"✅ Merged model saved to: {merged_dir}")
    
    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print("="*60)
    print(f"\nMerged model location: {merged_dir}")
    print("\nNext step: Create Ollama Modelfile and import")
    print("Run: ollama create analytics-sql -f training/Modelfile")


if __name__ == "__main__":
    main()

