"""
Encodes meme cognitive rationales into 768-dimensional CLIP text representations.
Saves features as .pt embeddings for Knowledge-Gated Fusion (K-CARN).
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Encode rationales using CLIP text encoder")
    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--dataset", type=str, default="FB", choices=["FB", "HarMeme"])
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def encode_split(split_file, text_model, tokenizer, batch_size=64):
    """Reads a rationales jsonl file and returns tensor of pooled features and ids."""
    ids = []
    rationales = []
    labels = []
    
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ids.append(str(item["id"]))
            rationales.append(item.get("rationale", item.get("text", "")))
            lbl = item.get("label", 0)
            if lbl is None:
                lbl = 0
            labels.append(int(lbl))

    all_feats = []
    total = len(rationales)
    
    for i in tqdm(range(0, total, batch_size), desc=f"Encoding {os.path.basename(split_file)}"):
        batch_texts = rationales[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = text_model(**inputs)
            # pooler_output is 768-d
            pooled = outputs.pooler_output.cpu()
            all_feats.append(pooled)

    feats_tensor = torch.cat(all_feats, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return ids, feats_tensor, labels_tensor


def main(args):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    rat_dir = os.path.join(base_dir, "data", "rationales", args.dataset)
    save_dir = os.path.join(base_dir, "data", "Knowledge_Embedding", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Initializing CLIP Text Encoder: {args.model} on {device.upper()}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    text_model = CLIPTextModel.from_pretrained(args.model).to(device).eval()

    splits = {
        "FB": ["train", "dev_seen", "test_seen", "test_unseen"],
        "HarMeme": ["train", "dev", "test"]
    }[args.dataset]

    for split in splits:
        in_file = os.path.join(rat_dir, f"{split}.jsonl")
        if not os.path.exists(in_file):
            print(f"[SKIP] Not found: {in_file}")
            continue

        ids, feats, labels = encode_split(in_file, text_model, tokenizer, batch_size=args.batch_size)
        
        save_name = f"{split}_openai_clip-vit-large-patch14-336_HF.pt"
        save_path = os.path.join(save_dir, save_name)
        
        torch.save({
            "ids": ids,
            "knowledge_feats": feats,
            "labels": labels
        }, save_path)
        
        print(f"[SAVED] {split}: shape={feats.shape} -> {save_path}")

    print(f"\n[DONE] All {args.dataset} knowledge embeddings saved successfully!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
