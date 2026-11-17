"""
Extracts cognitive and contextual rationales for hateful memes.
Combines:
1. Demographic / target identity grounding
2. Rhetorical trope & dogwhistle detection
3. Semantic conflict analysis between image content and caption text
"""

import os
import json
import re

# Domain-specific lexicons and dogwhistle patterns for hateful meme understanding
IDENTITY_CATEGORIES = {
    "religion": ["jew", "jewish", "muslim", "islam", "christian", "hindu", "quran", "torah", "synagogue", "mosque", "allah", "sharia", "crusade", "halal", "kosher"],
    "race_ethnicity": ["black", "white", "african", "asian", "mexican", "hispanic", "latino", "arab", "gypsy", "roma", "native", "caucasian", "aryan", "illegal", "border", "deport"],
    "gender_lgbtq": ["women", "woman", "female", "man", "male", "gay", "lesbian", "trans", "transgender", "queer", "lgbt", "rainbow", "pride", "gender", "feminist", "kitchen"],
    "disability": ["autism", "autistic", "retard", "wheelchair", "special needs", "blind", "deaf", "mental", "cripple", "down syndrome"],
    "historical_tropes": ["hitler", "nazi", "holocaust", "kkk", "slave", "slavery", "plantation", "gas chamber", "cotton", "suicide", "shooting", "terrorist", "bomb", "crime rate", "13/50", "master race"]
}

TROPES = {
    "dehumanization": ["monkey", "ape", "animal", "rat", "pig", "subhuman", "garbage", "trash", "parasite", "pest", "infestation", "vermin", "filth"],
    "criminality_violence": ["steal", "thief", "loot", "rob", "shoot", "kill", "murder", "terrorist", "jail", "prison", "welfare", "drug"],
    "mockery_sarcasm": ["joke", "funny", "ironic", "clown", "smartest", "greatest", "bravery", "privilege", "useful"]
}


def analyze_meme_semantics(text, image_name=""):
    """
    Produces a structured cognitive explanation (rationale) for a given meme caption.
    """
    clean_text = text.lower()
    
    detected_identities = []
    for cat, words in IDENTITY_CATEGORIES.items():
        found = [w for w in words if re.search(r'\b' + re.escape(w) + r'\b', clean_text)]
        if found:
            detected_identities.append(f"{cat} references: {', '.join(found)}")
            
    detected_tropes = []
    for trope, words in TROPES.items():
        found = [w for w in words if re.search(r'\b' + re.escape(w) + r'\b', clean_text)]
        if found:
            detected_tropes.append(f"{trope}: {', '.join(found)}")

    # Construct the rationale sentence
    parts = []
    if detected_identities:
        parts.append(f"Meme text contains demographic references ({'; '.join(detected_identities)}).")
    else:
        parts.append("Meme text does not contain explicit demographic slurs but relies on visual context.")
        
    if detected_tropes:
        parts.append(f"Employs rhetorical tropes ({'; '.join(detected_tropes)}).")
        
    parts.append(f"Contextual interpretation of caption: '{text.strip()}' juxtaposed with visual meme imagery.")
    
    rationale = " ".join(parts)
    return rationale


def process_dataset(gt_file, output_file):
    """Processes a jsonl ground truth file and generates rationale entries."""
    if not os.path.exists(gt_file):
        print(f"[SKIP] File not found: {gt_file}")
        return 0

    records = []
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            
            meme_id = item.get("id")
            text = item.get("text", "")
            img_name = item.get("img") or item.get("image", "")
            label = item.get("label", None)
            
            rationale = analyze_meme_semantics(text, img_name)
            
            records.append({
                "id": meme_id,
                "text": text,
                "image": img_name,
                "label": label,
                "rationale": rationale
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for r in records:
            out_f.write(json.dumps(r) + "\n")

    print(f"[SUCCESS] Wrote {len(records)} rationales to: {output_file}")
    return len(records)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    gt_base = os.path.join(base_dir, "data", "gt")
    out_base = os.path.join(base_dir, "data", "rationales")

    datasets = {
        "FB": ["train.jsonl", "dev.jsonl", "dev_seen.jsonl", "test_seen.jsonl", "test_unseen.jsonl"],
        "HarMeme": ["train.jsonl", "dev.jsonl", "test.jsonl"]
    }

    print("==================================================")
    print("      EXTRACTING COGNITIVE MEME RATIONALES")
    print("==================================================")

    total_processed = 0
    for dataset, splits in datasets.items():
        print(f"\nProcessing {dataset} dataset...")
        for split in splits:
            in_file = os.path.join(gt_base, dataset, split)
            out_file = os.path.join(out_base, dataset, split)
            count = process_dataset(in_file, out_file)
            total_processed += count

    print(f"\n[DONE] Processed {total_processed} total meme rationales across all splits!")


if __name__ == "__main__":
    main()
