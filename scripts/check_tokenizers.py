#!/usr/bin/env python3
"""
check_tokenizers.py
───────────────────
Türkçe LLM modellerinin tokenizer'larını karşılaştırır.
vocab_size, special_tokens ve padding_side kontrolü yapar.
Her model çifti için merge edilebilirlik durumunu döndürür.

Kullanım:
    python check_tokenizers.py
    python check_tokenizers.py --models model1 model2
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


# Varsayılan modeller (Mistral-7b-tr HF'den kaldırıldı)
DEFAULT_MODELS = [
    "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
    "Trendyol/Trendyol-LLM-8b-chat-v2.0",
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
        "full_vocab_size": len(tokenizer),
        "padding_side": getattr(tokenizer, "padding_side", "N/A"),
        "special_tokens": special_tokens,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_max_length": getattr(tokenizer, "model_max_length", "N/A"),
    }


def check_compatibility(info_a: dict, info_b: dict) -> dict:
    """İki tokenizer arasındaki uyumu kontrol eder."""
    issues = []      # Gerçek blocker'lar
    warnings = []    # Merge'i engellemez ama dikkat edilmeli

    # ── Gerçek blocker: vocab size farkı ──
    vocab_diff = abs(info_a["full_vocab_size"] - info_b["full_vocab_size"])
    vocab_match = info_a["full_vocab_size"] == info_b["full_vocab_size"]
    if not vocab_match:
        issues.append(
            f"❌ Vocab boyutu farklı: {info_a['model'].split('/')[-1]}={info_a['full_vocab_size']}, "
            f"{info_b['model'].split('/')[-1]}={info_b['full_vocab_size']} (fark: {vocab_diff}) — merge edilemez"
        )

    # ── Gerçek blocker: tokenizer sınıfı farkı ──
    class_match = info_a["tokenizer_class"] == info_b["tokenizer_class"]
    if not class_match:
        issues.append(
            f"❌ Tokenizer sınıfı farklı: {info_a['model'].split('/')[-1]}={info_a['tokenizer_class']}, "
            f"{info_b['model'].split('/')[-1]}={info_b['tokenizer_class']}"
        )

    # ── Uyarı (blocker değil): eos_token farkı ──
    eos_a = info_a["special_tokens"].get("eos_token")
    eos_b = info_b["special_tokens"].get("eos_token")
    if eos_a != eos_b:
        warnings.append(
            f"⚠️  eos_token farklı: '{eos_a}' vs '{eos_b}' "
            f"(merge'i engellemez, chat template farkı)"
        )

    # ── Uyarı (blocker değil): bos_token farkı ──
    bos_a = info_a["special_tokens"].get("bos_token")
    bos_b = info_b["special_tokens"].get("bos_token")
    if bos_a != bos_b:
        warnings.append(
            f"⚠️  bos_token farklı: '{bos_a}' vs '{bos_b}' "
            f"(merge'i engellemez)"
        )

    compatible = len(issues) == 0
    return {
        "model_a": info_a["model"],
        "model_b": info_b["model"],
        "compatible": compatible,
        "vocab_match": vocab_match,
        "class_match": class_match,
        "issues": issues,
        "warnings": warnings,
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
        help="Kontrol edilecek model isimleri (varsayılan: 2 Türkçe model)",
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
            info["full_vocab_size"],
            info["tokenizer_class"],
            info["padding_side"],
            special_str,
        ])

    headers = ["Model", "Vocab Size", "Full Vocab", "Tokenizer Sınıfı",
               "Padding", "Özel Tokenler"]
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
        status = "✅ Uyumlu (merge edilebilir)" if result["compatible"] else "❌ Uyumsuz"

        print(f"\n  {short_a}  ↔  {short_b}:  {status}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"    {issue}")
        if result["warnings"]:
            for warning in result["warnings"]:
                print(f"    {warning}")

    # ── 4) Merge edilebilirlik özeti ─────────────────────────────
    print("\n" + "─" * 70)
    print("📋 MERGE EDİLEBİLİRLİK ÖZETİ")
    print("─" * 70)

    merge_table = []
    for result in compatibility_results:
        short_a = result["model_a"].split("/")[-1]
        short_b = result["model_b"].split("/")[-1]
        all_notes = result["issues"] + result["warnings"]
        merge_table.append([
            f"{short_a} + {short_b}",
            "✅ True" if result["compatible"] else "❌ False",
            "; ".join(all_notes) if all_notes else "—",
        ])

    print(tabulate(merge_table,
                   headers=["Model Çifti", "Merge Edilebilir?", "Notlar"],
                   tablefmt="grid"))

    # ── 5) Çıkarma önerisi ───────────────────────────────────────
    exclusion = suggest_exclusion(compatibility_results, infos)
    if exclusion:
        short_exc = exclusion.split("/")[-1]
        print(f"\n⚠️  ÖNERİ: '{short_exc}' modeli en çok uyumsuzluğa sahip.")
        print(f"   Bu modeli merge'den çıkarmayı düşünebilirsiniz.")
    else:
        print("\n✅ Tüm modeller birbiriyle uyumlu! Merge işlemine devam edilebilir.")

    # Uyarı varsa bilgilendir
    all_warnings = []
    for result in compatibility_results:
        all_warnings.extend(result["warnings"])
    if all_warnings:
        print(f"\nℹ️  {len(all_warnings)} uyarı var (bunlar merge'i engellemez):")
        for w in all_warnings:
            print(f"   {w}")

    print("\n" + "=" * 70)
    print("✅ Kontrol tamamlandı.")
    print("=" * 70)

    has_incompatible = any(not r["compatible"] for r in compatibility_results)
    return 1 if has_incompatible else 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
