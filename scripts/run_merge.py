#!/usr/bin/env python3
"""
run_merge.py
────────────
mergekit kullanarak Türkçe LLM modellerini birleştirir.
SLERP, TIES veya DARE stratejilerini destekler.

Kullanım:
    python run_merge.py --strategy slerp
    python run_merge.py --strategy ties --output ./my_model
    python run_merge.py --strategy dare --config configs/dare_config.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except ImportError:
    print("❌ Gerekli kütüphaneler bulunamadı. Lütfen kurun:")
    print("   pip install transformers torch")
    sys.exit(1)

from tqdm import tqdm

# ── Sabitler ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_CONFIG_MAP = {
    "slerp": PROJECT_ROOT / "configs" / "slerp_config.yaml",
    "ties": PROJECT_ROOT / "configs" / "ties_config.yaml",
    "dare": PROJECT_ROOT / "configs" / "dare_config.yaml",
}

STRATEGY_OUTPUT_MAP = {
    "slerp": "TR-Llama-8B-Cosmos-Trendyol_SLERP_v1",
    "ties": "TR-Llama-8B-3way_TIES_v1",
    "dare": "TR-Llama-8B-3way_DARE_v1",
}

# Sanity check için Türkçe test cümleleri
SANITY_PROMPTS = [
    "Türkiye'nin başkenti neresidir?",
    "Yapay zeka nedir? Kısaca açıkla.",
    "İstanbul'un en ünlü tarihi yapıları hangileridir?",
]


def run_mergekit(config_path: str, output_path: str, strategy: str):
    """mergekit-yaml komutunu çalıştırır."""
    print(f"\n{'='*70}")
    print(f"🔀 MERGE BAŞLIYOR: {strategy.upper()}")
    print(f"{'='*70}")
    print(f"  📄 Config : {config_path}")
    print(f"  📂 Output : {output_path}")
    print()

    # mergekit-yaml komutu
    cmd = [
        sys.executable, "-m", "mergekit.scripts.run_yaml",
        str(config_path),
        str(output_path),
        "--copy-tokenizer",
        "--allow-crimes",
        "--lazy-unpickle",
    ]

    print(f"  🖥️  Komut: {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Çıktıyı gerçek zamanlı göster
        for line in process.stdout:
            print(f"  {line}", end="")

        process.wait()

        if process.returncode != 0:
            print(f"\n❌ mergekit hata kodu ile çıktı: {process.returncode}")
            return False

        print(f"\n✅ Merge tamamlandı: {output_path}")
        return True

    except FileNotFoundError:
        print("❌ mergekit bulunamadı. Lütfen kurun:")
        print("   pip install mergekit")
        return False
    except Exception as e:
        print(f"❌ Merge sırasında hata: {e}")
        return False


def sanity_check(model_path: str, strategy: str):
    """Merge sonrası basit Türkçe test cümleleri üretir."""
    print(f"\n{'─'*70}")
    print(f"🧪 SANİTY CHECK: {strategy.upper()} modeli")
    print(f"{'─'*70}")

    try:
        print("  📥 Model yükleniyor (4-bit quantized)...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("  ✅ Model yüklendi.\n")

        for i, prompt in enumerate(SANITY_PROMPTS, 1):
            print(f"  {'─'*60}")
            print(f"  📝 Soru {i}: {prompt}")

            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.1,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Sadece modelin ürettiği kısmı göster
            generated = response[len(prompt):].strip()
            print(f"  💬 Yanıt: {generated[:300]}")
            print()

        # Belleği temizle
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  ✅ Sanity check tamamlandı.\n")
        return True

    except Exception as e:
        print(f"  ❌ Sanity check hatası: {e}")
        print(f"  ℹ️  Model yine de kaydedilmiş olabilir: {model_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="mergekit ile Türkçe LLM modellerini birleştirir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run_merge.py --strategy slerp
  python run_merge.py --strategy ties
  python run_merge.py --strategy dare
  python run_merge.py --strategy slerp --output ./custom_output
  python run_merge.py --strategy ties --config ./my_config.yaml
        """,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["slerp", "ties", "dare"],
        help="Merge stratejisi: slerp, ties veya dare",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Özel config dosyası yolu (varsayılan: configs/{strateji}_config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Çıktı dizini (varsayılan: ./merged_models/{model_adı})",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Sanity check'i atla",
    )
    args = parser.parse_args()

    strategy = args.strategy

    # Config dosyasını belirle
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = STRATEGY_CONFIG_MAP.get(strategy)
        if config_path is None:
            print(f"❌ Bilinmeyen strateji: {strategy}")
            sys.exit(1)

    if not config_path.exists():
        print(f"❌ Config dosyası bulunamadı: {config_path}")
        sys.exit(1)

    # Output dizinini belirle
    if args.output:
        output_path = Path(args.output)
    else:
        model_name = STRATEGY_OUTPUT_MAP.get(strategy, f"merged_{strategy}")
        output_path = Path("./merged_models") / strategy

    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Merge işlemini çalıştır
    success = run_mergekit(str(config_path), str(output_path), strategy)

    if not success:
        print("\n❌ Merge başarısız oldu.")
        sys.exit(1)

    # Sanity check
    if not args.skip_sanity_check:
        sanity_check(str(output_path), strategy)
    else:
        print("\n⏭️  Sanity check atlandı (--skip-sanity-check).")

    print(f"\n{'='*70}")
    print(f"✅ {strategy.upper()} MERGE TAMAMLANDI")
    print(f"   Model kaydedildi: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
