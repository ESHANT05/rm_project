"""
High-performance, zero-dependency OCR engine for memes using Windows native OCR & PIL enhancement.
Extracts embedded meme captions directly from images.
"""
import os
import subprocess
import tempfile
import re
from PIL import Image, ImageEnhance, ImageFilter

PS_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ocr_test.ps1")


def _run_native_ocr(image_path):
    if not os.path.exists(PS_SCRIPT_PATH):
        return ""
    try:
        cmd = [
            "powershell", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", os.path.abspath(PS_SCRIPT_PATH), "-ImagePath", os.path.abspath(image_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        for line in proc.stdout.splitlines():
            if "EXTRACTED_TEXT:" in line:
                raw_text = line.split("EXTRACTED_TEXT:")[1].strip()
                # Clean up punctuation artifacts from low-res meme noise
                cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', raw_text)
                return cleaned.strip()
    except Exception:
        pass
    return ""


def extract_meme_text(image_input):
    """
    Extracts text from a meme image automatically using multi-pass OCR.
    Args:
        image_input: PIL.Image.Image or file path string
    Returns:
        tuple: (extracted_text: str, engine_used: str)
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    else:
        pil_img = image_input.convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
        pil_img.save(tmp1.name)
        raw_path = tmp1.name

    # Pass 1: Raw Image OCR
    text = _run_native_ocr(raw_path)

    # Pass 2: Contrast-enhanced grayscale OCR (ideal for white Impact font with black border)
    if len(text.strip().split()) < 3:
        try:
            gray = pil_img.convert("L")
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(2.0)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                enhanced.save(tmp2.name)
                enh_path = tmp2.name
            
            enhanced_text = _run_native_ocr(enh_path)
            try:
                os.remove(enh_path)
            except Exception:
                pass

            if len(enhanced_text.strip().split()) > len(text.strip().split()):
                text = enhanced_text
        except Exception:
            pass

    try:
        os.remove(raw_path)
    except Exception:
        pass

    clean_text = " ".join(text.split()).strip()
    engine = "Windows Native OCR (Multi-Pass)" if clean_text else "None"
    return clean_text, engine
