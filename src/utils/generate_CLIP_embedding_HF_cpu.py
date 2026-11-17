"""
CPU-safe version of CLIP embedding generator.
Completely disables CUDA to avoid GPU-lost errors.
"""
import os
# Disable CUDA completely before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import torch
import gc
import time
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel

import sys
sys.path.append('./src')
from data_loader.dataset import (
    get_Dataloader,
)
from utils.extract_CLIP_features import extract_clip_features_HF

device = "cpu"


def parse_args_sys(args_list=None):
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--EXP_FOLDER", type=str, default="./data/CLIP_Embedding",
        help="The path to save results.",
    )
    arg_parser.add_argument(
        "--model", type=str, default="openai/clip-vit-large-patch14-336",
        help="The clip model to use",
    )
    arg_parser.add_argument("--image_size", type=int, default=336)
    arg_parser.add_argument("--dataset", type=str, default="FB")
    arg_parser.add_argument("--batch_size", type=int, default=8)
    arg_parser.add_argument("--all", type=bool, default=False)
    args = arg_parser.parse_args()
    return args


def main(args):
    if not os.path.exists("{}/{}".format(args.EXP_FOLDER, args.dataset)):
        os.makedirs("{}/{}".format(args.EXP_FOLDER, args.dataset))

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Dataset: {args.dataset}, Batch size: {args.batch_size}")
    
    print("[INFO] Loading CLIP model...")
    Vision_model = CLIPVisionModel.from_pretrained(args.model)
    Text_model = CLIPTextModel.from_pretrained(args.model)
    preprocess = CLIPProcessor.from_pretrained(args.model)
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    
    Vision_model.eval()
    Text_model.eval()
    print("[INFO] Model loaded successfully.")

    # Pass device="cpu" explicitly to get_Dataloader
    if args.dataset == "FB":
        train, dev_seen, test_seen, test_unseen = get_Dataloader(
            preprocess,
            batch_size=args.batch_size,
            num_workers=0,
            train_batch_size=args.batch_size,
            device="cpu",
            image_size=args.image_size,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen, test_unseen]
        name_list = ["train", "dev_seen", "test_seen", "test_unseen"]
    elif args.dataset in ["HarMeme", "MMHS", "Propaganda", "Tamil", "HarmC", "HarmP", "MultiOFF"]:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess,
            batch_size=args.batch_size,
            num_workers=0,
            train_batch_size=args.batch_size,
            device="cpu",
            image_size=args.image_size,
            dataset=args.dataset,
        )
        loader_list = [test_seen, train, dev_seen]
        name_list = ["test_seen", "train", "dev_seen"]
    elif "Memotion" in args.dataset:
        train, dev_seen, test_seen = get_Dataloader(
            preprocess,
            batch_size=args.batch_size,
            num_workers=0,
            train_batch_size=args.batch_size,
            device="cpu",
            image_size=args.image_size,
            dataset=args.dataset,
        )
        loader_list = [train, dev_seen, test_seen]
        name_list = ["train", "dev_seen", "test_seen"]
    else:
        raise ValueError("Dataset not supported")

    total_start = time.time()
    
    for loader, name in zip(loader_list, name_list):
        split_start = time.time()
        print(f"\n[INFO] Processing split: {name} ({len(loader)} batches)")
        
        (
            all_img_feats,
            all_text_feats,
            pooler_img_feats,
            pooler_text_feats,
            labels,
            ids,
        ) = extract_clip_features_HF(loader, device, Vision_model, Text_model, preprocess, tokenizer, args.all)
        
        save_path = "{}/{}/{}_{}_HF.pt".format(
            args.EXP_FOLDER, args.dataset, name, str(args.model).replace("/", "_")
        )
        torch.save(
            {
                "ids": ids,
                "img_feats": pooler_img_feats,
                "text_feats": pooler_text_feats,
                "labels": labels,
            },
            save_path,
        )
        
        elapsed = time.time() - split_start
        print(f"[INFO] Split '{name}' done in {elapsed:.1f}s. Saved to {save_path}")
        
        del all_img_feats, all_text_feats, pooler_img_feats, pooler_text_feats, labels, ids
        gc.collect()

    total_elapsed = time.time() - total_start
    print(f"\n[INFO] All splits completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    args = parse_args_sys()
    print(args)
    main(args)
