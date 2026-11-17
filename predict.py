"""
Interactive inference script for RGCL / RAC Hateful Meme Detection.
Loads trained checkpoints and classifies memes with probability scores.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, CLIPTokenizer, CLIPTextModel

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from model.classifier import classifier_hateClipper


class DummyArgs:
    """Mock args required by classifier_hateClipper"""
    def __init__(self, dataset="FB"):
        self.dataset = dataset


def load_model(checkpoint_path, device="cuda", dataset="FB", map_dim=1024, proj_dim=1024, num_layers=3, fusion_mode="concat"):
    """Initializes classifier and loads trained weights."""
    image_dim = 1024  # CLIP-ViT-L-14/336 vision dim
    text_dim = 768    # CLIP-ViT-L-14/336 text dim
    dropout = [0.1, 0.4, 0.2]
    mock_args = DummyArgs(dataset=dataset)

    model = classifier_hateClipper(
        image_dim=image_dim,
        text_dim=text_dim,
        num_layers=num_layers,
        proj_dim=proj_dim,
        map_dim=map_dim,
        fusion_mode=fusion_mode,
        dropout=dropout,
        batch_norm=False,
        args=mock_args
    )

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class MemePredictor:
    def __init__(self, checkpoint_path, clip_model="openai/clip-vit-large-patch14-336", dataset="FB", fusion_mode="concat", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[INFO] Initializing MemePredictor on {self.device.upper()}...")
        print(f"[INFO] Loading CLIP encoder: {clip_model}...")
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model).to(self.device).eval()
        self.text_model = CLIPTextModel.from_pretrained(clip_model).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)

        print(f"[INFO] Loading trained hate classifier checkpoint ({fusion_mode} fusion)...")
        self.classifier = load_model(checkpoint_path, device=self.device, dataset=dataset, fusion_mode=fusion_mode)
        print(f"[SUCCESS] Ready for prediction!\n")

    @torch.no_grad()
    def predict(self, image_path, text_caption):
        """
        Classifies a single meme.
        Returns:
            prob: float (0.0 to 1.0) probability of being hateful
            is_hateful: bool
            label_text: str ('HATEFUL' or 'NOT HATEFUL')
        """
        image = Image.open(image_path).convert("RGB")
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        txt_inputs = self.tokenizer(text_caption, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)

        img_feats = self.vision_model(**img_inputs).pooler_output
        txt_feats = self.text_model(**txt_inputs).pooler_output

        logits = self.classifier(img_feats, txt_feats)
        prob = torch.sigmoid(logits).item()
        is_hateful = prob >= 0.5
        label_text = "HATEFUL" if is_hateful else "NOT HATEFUL (Benign)"

        return {
            "is_hateful": is_hateful,
            "label": label_text,
            "confidence": prob if is_hateful else (1.0 - prob),
            "hateful_probability": prob
        }


def run_demo(predictor, gt_file, img_base_dir, n_samples=5):
    """Runs prediction on random samples from the dataset and compares with ground truth."""
    print(f"==================================================")
    print(f"        RUNNING DEMO EVALUATION ON {n_samples} SAMPLES")
    print(f"==================================================\n")

    correct = 0
    total = 0

    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines[:n_samples]):
        item = json.loads(line)
        img_rel = item.get("img") or item.get("image")
        text = item.get("text", "")
        ground_truth = item.get("label", None)
        if ground_truth is None and "labels" in item:
            ground_truth = 1 if "harmful" in str(item["labels"]).lower() else 0

        # Construct image path
        img_path = os.path.join(img_base_dir, os.path.basename(img_rel))
        if not os.path.exists(img_path):
            # check direct path
            img_path = os.path.join(img_base_dir, img_rel)

        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {img_path}")
            continue

        res = predictor.predict(img_path, text)
        gt_text = "HATEFUL" if ground_truth == 1 else "NOT HATEFUL"
        is_correct = (res["is_hateful"] == bool(ground_truth))
        if is_correct:
            correct += 1
        total += 1

        print(f"--- Sample #{idx+1} ---")
        print(f"Image   : {os.path.basename(img_path)}")
        print(f"Text    : \"{text.strip()}\"")
        print(f"Ground Truth : {gt_text}")
        print(f"Predicted    : {res['label']} ({res['hateful_probability']*100:.1f}% hate prob)")
        print(f"Result       : {'✅ CORRECT' if is_correct else '❌ WRONG'}\n")

    if total > 0:
        print(f"Summary: {correct}/{total} correct ({correct/total*100:.1f}% accuracy on this mini-batch)")


def main():
    parser = argparse.ArgumentParser(description="Predict hatefulness of a meme")
    default_fb_ckpt = os.path.join(
        os.path.dirname(__file__),
        "rgcl_trained_models_and_results",
        "Retrieval", "FB", "RAC_TEST",
        "RAC_lr0.0001_Bz128_Ep5_cosSim_naive_drop[0.1, 0.4, 0.2]_topK5__PseudoGold_positive_1_hard_negative_1_seed0",
        "ckpt", "best_model_2_0.664.pt"
    )
    parser.add_argument("--checkpoint", type=str, default=default_fb_ckpt, help="Path to trained .pt checkpoint")
    parser.add_argument("--dataset", type=str, default="FB", choices=["FB", "HarMeme"])
    parser.add_argument("--fusion_mode", type=str, default="concat", choices=["concat", "cross_attention"], help="Modality fusion architecture")
    parser.add_argument("--demo", action="store_true", help="Run demo on samples from dev set")
    parser.add_argument("--samples", type=int, default=5, help="Number of demo samples")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--text", type=str, default=None, help="Text caption for single image")
    args = parser.parse_args()

    predictor = MemePredictor(checkpoint_path=args.checkpoint, dataset=args.dataset, fusion_mode=args.fusion_mode)

    if args.demo or (args.image is None and args.text is None):
        if args.dataset == "FB":
            gt_file = os.path.join(os.path.dirname(__file__), "data", "gt", "FB", "dev_seen.jsonl")
            img_dir = os.path.join(os.path.dirname(__file__), "data", "image", "FB", "All")
        else:
            gt_file = os.path.join(os.path.dirname(__file__), "data", "gt", "HarMeme", "dev.jsonl")
            img_dir = os.path.join(os.path.dirname(__file__), "data", "image", "HarMeme", "All")
            
        run_demo(predictor, gt_file, img_dir, n_samples=args.samples)
    else:
        if args.image and args.text:
            res = predictor.predict(args.image, args.text)
            print(f"\n================ Prediction ================")
            print(f"Meme Image : {args.image}")
            print(f"Meme Text  : {args.text}")
            print(f"Prediction : {res['label']}")
            print(f"Confidence : {res['confidence']*100:.2f}%")
            print(f"Hate Prob  : {res['hateful_probability']:.4f}")
            print(f"============================================")
        else:
            print("Please provide both --image and --text to classify a custom meme.")


if __name__ == "__main__":
    main()

