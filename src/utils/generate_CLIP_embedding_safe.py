"""
GPU-safe CLIP embedding generator with thermal protection.
Uses small batch size and cooling pauses to prevent RTX 3050 4GB crashes.
"""
import argparse
import torch
import gc
import time
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel

import sys
import os

sys.path.append('./src')
from data_loader.dataset import (
    get_Dataloader,
)
from utils.extract_CLIP_features import extract_clip_features_HF

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args_sys():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--EXP_FOLDER", type=str, default="./data/CLIP_Embedding")
    arg_parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336")
    arg_parser.add_argument("--image_size", type=int, default=336)
    arg_parser.add_argument("--dataset", type=str, default="FB")
    arg_parser.add_argument("--batch_size", type=int, default=4)
    arg_parser.add_argument("--all", type=bool, default=False)
    args = arg_parser.parse_args()
    return args


def main(args):
    if not os.path.exists("{}/{}".format(args.EXP_FOLDER, args.dataset)):
        os.makedirs("{}/{}".format(args.EXP_FOLDER, args.dataset))

    print(f"[INFO] Device: {device}", flush=True)
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
    print(f"[INFO] Dataset: {args.dataset}, Batch size: {args.batch_size}", flush=True)
    
    print("[INFO] Loading CLIP model...", flush=True)
    Vision_model = CLIPVisionModel.from_pretrained(args.model)
    Text_model = CLIPTextModel.from_pretrained(args.model)
    preprocess = CLIPProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    
    if device == "cuda":
        Vision_model.cuda().eval()
        Text_model.cuda().eval()
    else:
        Vision_model.eval()
        Text_model.eval()
    print("[INFO] Model loaded.", flush=True)

    # num_workers=0 on Windows to avoid multiprocessing issues
    if args.dataset == "FB":
        train, dev_seen, test_seen, test_unseen = get_Dataloader(
            preprocess, batch_size=args.batch_size, num_workers=0,
            train_batch_size=args.batch_size, device=device,
            image_size=args.image_size, dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen, test_unseen]
        name_list = ["train", "dev_seen", "test_seen", "test_unseen"]
    elif args.dataset in ["HarMeme", "MMHS", "Propaganda", "Tamil", "HarmC", "HarmP", "MultiOFF"]:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess, batch_size=args.batch_size, num_workers=0,
            train_batch_size=args.batch_size, device=device,
            image_size=args.image_size, dataset=args.dataset,
        )
        loader_list = [test_seen, train, dev_seen]
        name_list = ["test_seen", "train", "dev_seen"]
    elif "Memotion" in args.dataset:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess, batch_size=args.batch_size, num_workers=0,
            train_batch_size=args.batch_size, device=device,
            image_size=args.image_size, dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen]
        name_list = ["train", "dev_seen", "test_seen"]
    else:
        raise ValueError("Dataset not supported")

    total_start = time.time()
    
    for loader, name in zip(loader_list, name_list):
        split_start = time.time()
        num_batches = len(loader)
        print(f"\n[INFO] Processing split: {name} ({num_batches} batches)", flush=True)
        
        # Clear GPU cache before each split
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        (
            all_img_feats, all_text_feats,
            pooler_img_feats, pooler_text_feats,
            labels, ids,
        ) = extract_clip_features_HF(
            loader, device, Vision_model, Text_model, preprocess, tokenizer, args.all
        )
        
        save_path = "{}/{}/{}_{}_HF.pt".format(
            args.EXP_FOLDER, args.dataset, name, str(args.model).replace("/", "_")
        )
        torch.save({
            "ids": ids,
            "img_feats": pooler_img_feats,
            "text_feats": pooler_text_feats,
            "labels": labels,
        }, save_path)
        
        elapsed = time.time() - split_start
        print(f"[INFO] Split '{name}' done in {elapsed:.1f}s. Saved: {save_path}", flush=True)
        
        # Cleanup between splits
        del all_img_feats, all_text_feats, pooler_img_feats, pooler_text_feats, labels, ids
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        # Cool-down pause between splits (15 seconds)
        if device == "cuda":
            print("[INFO] Cooling pause (15s)...", flush=True)
            time.sleep(15)

    total_elapsed = time.time() - total_start
    print(f"\n[INFO] ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    args = parse_args_sys()
    print(args, flush=True)
    main(args)
