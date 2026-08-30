"""
One-time conversion of NLLB-200 to CTranslate2 format.

Install deps first (temporary, can uninstall torch after):
    pip install torch transformers ctranslate2 sentencepiece

Then run:
    python convert_model.py

After conversion torch is no longer needed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR = Path.home() / ".cache" / "live_translate" / "nllb-ct2"


def find_converter() -> str:
    cmd = shutil.which("ct2-transformers-converter")
    if cmd:
        return cmd
    # Windows: look next to the python executable
    scripts = Path(sys.executable).parent / "ct2-transformers-converter.exe"
    if scripts.exists():
        return str(scripts)
    raise RuntimeError(
        "ct2-transformers-converter not found.\n"
        "Install: pip install ctranslate2"
    )


def convert_weights():
    if (OUTPUT_DIR / "config.json").exists():
        print("Weights already converted, skipping.")
        return
    # Remove incomplete dir so converter doesn't complain
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    converter = find_converter()
    print(f"Converting {MODEL_ID} → int8 CTranslate2...")
    subprocess.run(
        [
            converter,
            "--model", MODEL_ID,
            "--output_dir", str(OUTPUT_DIR),
            "--quantization", "int8",
        ],
        check=True,
    )


def save_tokenizer():
    # newer transformers writes tokenizer.json, older ones sentencepiece.bpe.model
    if any((OUTPUT_DIR / name).exists()
           for name in ("tokenizer.json", "sentencepiece.bpe.model")):
        print("Tokenizer already saved, skipping.")
        return
    from transformers import NllbTokenizer
    print("Saving tokenizer...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = NllbTokenizer.from_pretrained(MODEL_ID)
    tok.save_pretrained(str(OUTPUT_DIR))


def main():
    print(f"Output: {OUTPUT_DIR}\n")

    convert_weights()
    save_tokenizer()

    size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"\nDone! Model size: {size_mb:.0f} MB")
    print("You can now uninstall torch:  pip uninstall torch")
    print("Start translator:             python main.py")


if __name__ == "__main__":
    main()
