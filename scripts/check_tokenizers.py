#!/usr/bin/env python3
"""
check_tokenizers.py
───────────────────
Üç Türkçe LLM modelinin tokenizer'larını karşılaştırır.
vocab_size, special_tokens ve padding_side kontrolü yapar.
Her model çifti için merge edilebilirlik durumunu döndürür.

Kullanım:
    python check_tokenizers.py
    python check_tokenizers.py --models model1 model2 model3
"""

import argparse
import sys
from itertools import combinations

from tabulate import tabulate

try:
    from transformers import AutoTokenizer
except ImportError:
    print("❌ transformers kütüphanesi bulunamadı. Lütfen kurun:")
    print("   pip install transformers sentencepiece protobuf")
    sys.exit(1)


# Varsayılan modeller
DEFAULT_MODELS = [
    "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
    "Trendyol/Trendyol-LLM-8b-chat-v2.0",
    "malhajar/Mistral-7b-tr",
]


def load_tokenizer(model_name: str):
    """Tokenizer'ı yükler, hata durumunda None döner."""
    try:
        print(f"  📥 Yükleniyor: {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  ✅ Yüklendi: {model_name}")
        return tokenizer
    except Exception as e:
        print(f"  ❌ Yüklenemedi: {model_name} → {e}")
        return None


def get_tokenizer_info(tokenizer, model_name: str) -> dict:
    """Tokenizer'dan bilgi çıkarır."""
    special_tokens = {}
    for attr in ["bos_token", "eos_token", "pad_token", "unk_token",
                 "sep_token", "cls_token", "mask_token"]:
        val = getattr(tokenizer, attr, None)
        if val is not None:
            special_tokens[attr] = val

    return {
        "model": model_name,
        "vocab_size": tokenizer.vocab_size,
        "padding_side": getattr(tokenizer, "padding_side", "N/A"),
        "special_tokens": special_tokens,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_max_length": getattr(tokenizer, "model_max_length", "N/A"),
    }


def check_compatibility(info_a: dict, info_b: dict) -> dict:
    """İki tokenizer arasındaki uyumu kontrol eder."""
    issues = []

    # Vocab size kontrolü
    vocab_diff = abs(info_a["vocab_size"] - info_b["vocab_size"])
    vocab_match = info_a["vocab_size"] == info_b["vocab_size"]
    if not vocab_match:
        issues.append(
            f"Vocab boyutu farklı: {info_a['model']}={info_a['vocab_size']}, "
            f"{info_b['model']}={info_b['vocab_size']} (fark: {vocab_diff})"
        )

    # Tokenizer sınıfı kontrolü
    class_match = info_a["tokenizer_class"] == info_b["tokenizer_class"]
    if not class_match:
        issues.append(
            f"Tokenizer sınıfı farklı: {info_a['model']}={info_a['tokenizer_class']}, "
            f"{info_b['model']}={info_b['tokenizer_class']}"
        )

    # BOS/EOS token kontrolü
    for token_name in ["bos_token", "eos_token"]:
        tok_a = info_a["special_tokens"].get(token_name)
        tok_b = info_b["special_tokens"].get(token_name)
        if tok_a != tok_b:
            issues.append(
                f"{token_name} farklı: {info_a['model']}='{tok_a}', "
                f"{info_b['model']}='{tok_b}'"
            )

    compatible = len(issues) == 0
    return {
        "model_a": info_a["model"],
        "model_b": info_b["model"],
        "compatible": compatible,
        "vocab_match": vocab_match,
        "class_match": class_match,
        "issues": issues,
    }


def suggest_exclusion(compatibility_results: list, model_infos: list):
    """Uyumsuz model varsa hangisini çıkarmak gerektiğini önerir."""
    incompatible_models = {}
    for result in compatibility_results:
        if not result["compatible"]:
            for model in [result["model_a"], result["model_b"]]:
                incompatible_models[model] = incompatible_models.get(model, 0) + 1

    if not incompatible_models:
        return None

    # En çok uyumsuzluğa sahip modeli öner
    worst_model = max(incompatible_models, key=incompatible_models.get)
    return worst_model


def main():
    parser = argparse.ArgumentParser(
        description="Türkçe LLM modellerinin tokenizer uyumluluğunu kontrol eder."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Kontrol edilecek model isimleri (varsayılan: 3 Türkçe model)",
    )
    args = parser.parse_args()

    models = args.models
    print("=" * 70)
    print("🔍 TOKENIZER UYUMLULUK KONTROLÜ")
    print("=" * 70)
    print(f"\nKontrol edilecek modeller ({len(models)} adet):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")
    print()

    # ── 1) Tokenizer'ları yükle ──────────────────────────────────
    print("─" * 70)
    print("📥 Tokenizer'lar yükleniyor...")
    print("─" * 70)
    tokenizers = {}
    infos = []
    for model_name in models:
        tok = load_tokenizer(model_name)
        if tok is None:
            print(f"\n⚠️  {model_name} yüklenemedi, atlanıyor.\n")
            continue
        tokenizers[model_name] = tok
        infos.append(get_tokenizer_info(tok, model_name))

    if len(infos) < 2:
        print("\n❌ En az 2 model gerekli. Çıkılıyor.")
        sys.exit(1)

    # ── 2) Bilgileri tablo olarak göster ─────────────────────────
    print("\n" + "─" * 70)
    print("📊 TOKENIZER BİLGİLERİ")
    print("─" * 70)

    table_data = []
    for info in infos:
        short_name = info["model"].split("/")[-1]
        special_str = ", ".join(
            f"{k}='{v}'" for k, v in list(info["special_tokens"].items())[:4]
        )
        table_data.append([
            short_name,
            info["vocab_size"],
            info["tokenizer_class"],
            info["padding_side"],
            info["model_max_length"],
            special_str,
        ])

    headers = ["Model", "Vocab Size", "Tokenizer Sınıfı", "Padding",
               "Max Length", "Özel Tokenler"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # ── 3) Çift çift uyumluluk kontrolü ─────────────────────────
    print("\n" + "─" * 70)
    print("🔗 UYUMLULUK KONTROLÜ (her çift için)")
    print("─" * 70)

    compatibility_results = []
    for info_a, info_b in combinations(infos, 2):
        result = check_compatibility(info_a, info_b)
        compatibility_results.append(result)

        short_a = info_a["model"].split("/")[-1]
        short_b = info_b["model"].split("/")[-1]
        status = "✅ Uyumlu" if result["compatible"] else "❌ Uyumsuz"

        print(f"\n  {short_a}  ↔  {short_b}:  {status}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"    ⚠️  {issue}")

    # ── 4) Merge edilebilirlik özeti ─────────────────────────────
    print("\n" + "─" * 70)
    print("📋 MERGE EDİLEBİLİRLİK ÖZETİ")
    print("─" * 70)

    merge_table = []
    for result in compatibility_results:
        short_a = result["model_a"].split("/")[-1]
        short_b = result["model_b"].split("/")[-1]
        merge_table.append([
            f"{short_a} + {short_b}",
            "✅ True" if result["compatible"] else "❌ False",
            "; ".join(result["issues"]) if result["issues"] else "—",
        ])

    print(tabulate(merge_table,
                   headers=["Model Çifti", "Merge Edilebilir?", "Sorunlar"],
                   tablefmt="grid"))

    # ── 5) Çıkarma önerisi ───────────────────────────────────────
    exclusion = suggest_exclusion(compatibility_results, infos)
    if exclusion:
        short_exc = exclusion.split("/")[-1]
        print(f"\n⚠️  ÖNERİ: '{short_exc}' modeli en çok uyumsuzluğa sahip.")
        print(f"   Bu modeli merge'den çıkarmayı düşünebilirsiniz.")
        print(f"   (Tam ad: {exclusion})")
    else:
        print("\n✅ Tüm modeller birbiriyle uyumlu görünüyor!")

    print("\n" + "=" * 70)
    print("✅ Kontrol tamamlandı.")
    print("=" * 70)

    # Sonuç kodu
    has_incompatible = any(not r["compatible"] for r in compatibility_results)
    return 1 if has_incompatible else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
