#!/usr/bin/env python3
"""
benchmark.py
────────────
Merge edilmiş ve baseline Türkçe LLM modellerini değerlendirir.

Metrikler:
  1. Türkçe perplexity (mc4/tr veya cc100/tr, ilk 500 örnek)
  2. 20 Türkçe soruya yanıt kalitesi

Kullanım:
    python benchmark.py --model all
    python benchmark.py --model ./merged_models/slerp
    python benchmark.py --model ./merged_models/ties --output results/my_results.json
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from tabulate import tabulate
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ transformers kütüphanesi bulunamadı.")
    print("   pip install transformers torch")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("❌ datasets kütüphanesi bulunamadı.")
    print("   pip install datasets")
    sys.exit(1)

# ── Sabitler ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "benchmark_results.json"

BASELINE_MODEL = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"

MODEL_PATHS = {
    "slerp": "./merged_models/slerp",
    "ties": "./merged_models/ties",
    "dare": "./merged_models/dare",
    "baseline": BASELINE_MODEL,
}

# ── 20 Türkçe Soru Seti ──────────────────────────────────────────
TURKISH_QUESTIONS = {
    # 5 Genel Bilgi (Türkiye tarihi, coğrafya)
    "genel_bilgi": [
        {
            "id": "gb1",
            "question": "Türkiye Cumhuriyeti ne zaman kurulmuştur? Kısaca açıklayın.",
            "expected_keywords": ["1923", "29 Ekim", "Atatürk"],
        },
        {
            "id": "gb2",
            "question": "Türkiye'nin en uzun nehri hangisidir ve nereye dökülür?",
            "expected_keywords": ["Kızılırmak", "Karadeniz"],
        },
        {
            "id": "gb3",
            "question": "İstanbul Boğazı hangi denizleri birbirine bağlar?",
            "expected_keywords": ["Karadeniz", "Marmara"],
        },
        {
            "id": "gb4",
            "question": "Osmanlı İmparatorluğu'nun kurucusu kimdir ve hangi yılda kurulmuştur?",
            "expected_keywords": ["Osman", "1299"],
        },
        {
            "id": "gb5",
            "question": "Kapadokya hangi illerimizde yer almaktadır? En az 2 il sayın.",
            "expected_keywords": ["Nevşehir", "Kayseri", "Aksaray", "Niğde", "Kırşehir"],
        },
    ],
    # 5 Matematik
    "matematik": [
        {
            "id": "mat1",
            "question": "125'in küp kökü kaçtır?",
            "expected_keywords": ["5"],
        },
        {
            "id": "mat2",
            "question": "Bir üçgenin iç açıları toplamı kaç derecedir?",
            "expected_keywords": ["180"],
        },
        {
            "id": "mat3",
            "question": "15 × 24 işleminin sonucu kaçtır?",
            "expected_keywords": ["360"],
        },
        {
            "id": "mat4",
            "question": "Yarıçapı 7 cm olan bir dairenin alanı yaklaşık kaç cm²'dir? (π ≈ 3.14)",
            "expected_keywords": ["153", "154"],
        },
        {
            "id": "mat5",
            "question": "1'den 10'a kadar olan sayıların toplamı kaçtır?",
            "expected_keywords": ["55"],
        },
    ],
    # 5 Gramer/Dil
    "gramer": [
        {
            "id": "gr1",
            "question": "Türkçede kaç ünlü harf vardır? Sayınız.",
            "expected_keywords": ["8", "a", "e", "ı", "i", "o", "ö", "u", "ü"],
        },
        {
            "id": "gr2",
            "question": "'Gel-' fiilinin geniş zamanını (şimdiki zaman) 'ben' kişisiyle çekimleyin.",
            "expected_keywords": ["gelirim", "geliyorum"],
        },
        {
            "id": "gr3",
            "question": "'Kitap' kelimesinin çoğul halini ve belirtili nesne (-i hali) halini yazın.",
            "expected_keywords": ["kitaplar", "kitabı", "kitapları"],
        },
        {
            "id": "gr4",
            "question": "Büyük ünlü uyumunu kısaca açıklayın.",
            "expected_keywords": ["kalın", "ince", "ünlü", "uyum", "ek"],
        },
        {
            "id": "gr5",
            "question": "'Araba çok hızlı gidiyordu' cümlesinde özne ve yüklem nedir?",
            "expected_keywords": ["araba", "gidiyordu"],
        },
    ],
    # 5 Instruction Following
    "instruction": [
        {
            "id": "ins1",
            "question": "Türkiye'deki 3 büyük şehri madde halinde yazın.",
            "expected_keywords": ["İstanbul", "Ankara", "İzmir"],
        },
        {
            "id": "ins2",
            "question": "Sağlıklı yaşam için 3 öneri yazın. Her birini numaralı liste olarak verin.",
            "expected_keywords": ["1", "2", "3"],
        },
        {
            "id": "ins3",
            "question": "Aşağıdaki kelimeyi İngilizceye çevirin: 'bilgisayar'",
            "expected_keywords": ["computer"],
        },
        {
            "id": "ins4",
            "question": "'Yapay zeka' kavramını bir çocuğa anlatır gibi 2-3 cümle ile açıklayın.",
            "expected_keywords": ["bilgisayar", "öğren", "akıllı", "zeka"],
        },
        {
            "id": "ins5",
            "question": "Bir e-posta yazın: Konu 'Toplantı Hatırlatması'. Kısa ve resmi olsun.",
            "expected_keywords": ["toplantı", "sayın", "saygılarımla"],
        },
    ],
}


def compute_perplexity(model, tokenizer, num_samples: int = 200) -> float:
    """Türkçe veri seti üzerinde perplexity hesaplar."""
    print("    📊 Perplexity hesaplanıyor...")

    try:
        dataset = None
        dataset_name = None
        text_key = "text"

        # 1) wikipedia/tr (ücretsiz, geniş)
        try:
            dataset = load_dataset("wikipedia", "20220301.tr", split="train", streaming=True)
            dataset_name = "wikipedia/tr"
            text_key = "text"
        except Exception:
            pass

        # 2) wikiann/tr (ücretsiz NER veri seti, tokens alanı var)
        if dataset is None:
            try:
                dataset = load_dataset("wikiann", "tr", split="train", streaming=True)
                dataset_name = "wikiann/tr"
                text_key = "tokens"  # token listesi olarak gelir
            except Exception:
                pass

        # 3) Fallback: sabit Türkçe metinler
        if dataset is None:
            print("    ⚠️  Online veri seti yüklenemedi, dahili metinlerle hesaplanıyor.")
            dataset_name = "dahili_turkce"
            fallback_texts = [
                "Türkiye Cumhuriyeti 29 Ekim 1923 tarihinde Mustafa Kemal Atatürk önderliğinde kurulmuştur.",
                "İstanbul Boğazı Karadeniz ile Marmara Denizi arasında yer alan önemli bir su yoludur.",
                "Yapay zeka günümüzde birçok alanda kullanılmakta ve hızla gelişmektedir.",
                "Türk mutfağı dünyanın en zengin mutfaklarından biri olarak kabul edilmektedir.",
                "Kapadokya peri bacaları ve yeraltı şehirleriyle ünlü bir turizm bölgesidir.",
                "Ankara Türkiye Cumhuriyetinin başkenti olup ülkenin ikinci büyük şehridir.",
                "Osmanlı İmparatorluğu altı yüz yıldan fazla hüküm sürmüş büyük bir devlettir.",
                "Efes antik kenti Selçuk ilçesinde yer alan UNESCO Dünya Mirası listesindedir.",
                "Türkçe Ural Altay dil ailesine mensup sondan eklemeli bir dildir.",
                "Pamukkale travertenleri doğal güzellikleriyle her yıl milyonlarca turisti ağırlamaktadır.",
            ] * 20  # 200 örneğe çıkar
            dataset = [{"text": t} for t in fallback_texts]

        print(f"    📂 Veri seti: {dataset_name}")

        total_loss = 0.0
        total_tokens = 0
        processed = 0

        model.eval()
        with torch.no_grad():
            for sample in tqdm(dataset, total=num_samples, desc="    Perplexity"):
                if processed >= num_samples:
                    break

                # text alanını al
                if text_key == "tokens" and isinstance(sample.get("tokens"), list):
                    text = " ".join(sample["tokens"])
                else:
                    text = sample.get("text", "")

                if not text or len(text) < 20:
                    continue

                text = text[:512]

                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                input_ids = encodings["input_ids"].to(model.device)

                if input_ids.shape[1] < 2:
                    continue

                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

                num_tokens = input_ids.shape[1]
                total_loss += loss * num_tokens
                total_tokens += num_tokens
                processed += 1

        if total_tokens == 0:
            return float("nan")

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        print(f"    📊 Perplexity: {perplexity:.2f}")
        return round(perplexity, 2)

    except Exception as e:
        print(f"    ❌ Perplexity hesaplama hatası: {e}")
        return float("nan")


def score_response(response: str, expected_keywords: list) -> int:
    """Yanıtı beklenen anahtar kelimelerle puanlar (0 veya 1)."""
    response_lower = response.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    # En az 1 anahtar kelime eşleşmişse 1 puan
    return 1 if matched >= 1 else 0


def evaluate_questions(model, tokenizer) -> dict:
    """20 Türkçe soruyu modelle değerlendirir."""
    print("    📝 Türkçe soru değerlendirmesi...")

    responses = {}
    total_score = 0

    for category, questions in TURKISH_QUESTIONS.items():
        print(f"\n    📂 Kategori: {category}")
        for q in tqdm(questions, desc=f"    {category}"):
            try:
                inputs = tokenizer(
                    q["question"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        repetition_penalty=1.1,
                    )

                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Sadece üretilen kısmı al
                generated = full_response[len(q["question"]):].strip()

                sc = score_response(generated, q["expected_keywords"])
                total_score += sc

                responses[q["id"]] = {
                    "question": q["question"],
                    "response": generated[:500],
                    "score": sc,
                    "expected_keywords": q["expected_keywords"],
                }

            except Exception as e:
                responses[q["id"]] = {
                    "question": q["question"],
                    "response": f"HATA: {str(e)}",
                    "score": 0,
                    "expected_keywords": q["expected_keywords"],
                }

    return {"total_score": total_score, "max_score": 20, "responses": responses}


def evaluate_model(model_path: str, model_key: str) -> dict:
    """Tek bir modeli değerlendirir."""
    print(f"\n{'='*70}")
    print(f"🏋️ MODEL DEĞERLENDİRMESİ: {model_key.upper()}")
    print(f"   Path: {model_path}")
    print(f"{'='*70}")

    try:
        print("  📥 Model yükleniyor...")
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

        print("  ✅ Model yüklendi.")

        # 1. Perplexity
        perplexity = compute_perplexity(model, tokenizer)

        # 2. Türkçe soru değerlendirmesi
        qa_results = evaluate_questions(model, tokenizer)

        # Belleği temizle
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "perplexity": perplexity,
            "manual_score": f"{qa_results['total_score']}/{qa_results['max_score']}",
            "manual_score_num": qa_results["total_score"],
            "responses": qa_results["responses"],
        }

    except Exception as e:
        print(f"  ❌ Model değerlendirme hatası: {e}")
        return {
            "perplexity": None,
            "manual_score": None,
            "manual_score_num": 0,
            "responses": {"error": str(e)},
        }


def print_summary_table(results: dict):
    """Sonuçları özet tablo olarak yazdırır."""
    print(f"\n{'='*70}")
    print("📊 BENCHMARK SONUÇLARI ÖZETİ")
    print(f"{'='*70}\n")

    table_data = []
    strategy_map = {
        "slerp": "SLERP",
        "ties": "TIES",
        "dare": "DARE",
        "baseline": "Baseline",
    }

    # En iyi perplexity ve skorları bul
    valid_perplexities = {
        k: v["perplexity"]
        for k, v in results.items()
        if v.get("perplexity") and not math.isnan(v["perplexity"])
    }
    valid_scores = {
        k: v.get("manual_score_num", 0)
        for k, v in results.items()
    }

    best_perplexity = min(valid_perplexities, key=valid_perplexities.get) if valid_perplexities else None
    best_score = max(valid_scores, key=valid_scores.get) if valid_scores else None

    for key in ["slerp", "ties", "dare", "baseline"]:
        if key not in results:
            continue

        r = results[key]
        ppl = r.get("perplexity")
        ppl_str = f"{ppl:.2f}" if ppl and not math.isnan(ppl) else "N/A"
        score_str = r.get("manual_score", "N/A")

        # En iyi mi?
        is_best_ppl = (key == best_perplexity)
        is_best_score = (key == best_score)

        if is_best_ppl and is_best_score:
            best_str = "⭐ PPL+Skor"
        elif is_best_ppl:
            best_str = "⭐ PPL"
        elif is_best_score:
            best_str = "⭐ Skor"
        else:
            best_str = ""

        table_data.append([
            strategy_map.get(key, key),
            ppl_str,
            score_str,
            best_str,
        ])

    headers = ["Model", "Perplexity ↓", "Manuel/20", "En İyi?"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Türkçe LLM modellerini benchmark ile değerlendirir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python benchmark.py --model all
  python benchmark.py --model ./merged_models/slerp
  python benchmark.py --model baseline
  python benchmark.py --model all --output results/my_results.json
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model yolu veya 'all' (tüm modeller)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Sonuç dosyası yolu (varsayılan: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Perplexity için kullanılacak örnek sayısı (varsayılan: 500)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.model.lower() == "all":
        print(f"\n{'='*70}")
        print("🏁 TÜM MODELLER DEĞERLENDİRİLİYOR")
        print(f"{'='*70}")

        for key, path in MODEL_PATHS.items():
            if key != "baseline" and not Path(path).exists():
                print(f"\n⚠️  {key} modeli bulunamadı: {path} → Atlanıyor.")
                continue
            results[key] = evaluate_model(path, key)
    elif args.model in MODEL_PATHS:
        key = args.model
        path = MODEL_PATHS[key]
        results[key] = evaluate_model(path, key)
    else:
        # Özel yol verilmiş
        model_name = Path(args.model).name
        results[model_name] = evaluate_model(args.model, model_name)

    # Sonuçları kaydet
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "models": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Sonuçlar kaydedildi: {output_path}")

    # Özet tabloyu yazdır
    print_summary_table(results)


if __name__ == "__main__":
    main()
