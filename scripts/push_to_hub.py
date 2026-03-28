#!/usr/bin/env python3
"""
push_to_hub.py
──────────────
Merge edilmiş Türkçe LLM modellerini HuggingFace Hub'a yükler.
Model card'ı otomatik oluşturur.

Kullanım:
    python push_to_hub.py \\
        --model_path ./merged_models/slerp \\
        --repo_id Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1 \\
        --strategy SLERP \\
        --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0"
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

try:
    from huggingface_hub import HfApi, login
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("❌ Gerekli kütüphaneler bulunamadı.")
    print("   pip install transformers huggingface_hub")
    sys.exit(1)

# ── Sabitler ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

STRATEGY_PARAMS = {
    "SLERP": "t=0.5, dtype=bfloat16",
    "TIES": "weights=[0.6, 0.4], density=0.7, normalize=true, dtype=bfloat16",
    "DARE": "weights=[0.6, 0.4], density=0.5, normalize=true, dtype=bfloat16",
}

MODEL_CARD_TEMPLATE = """---
language: tr
license: llama3
tags:
  - turkish
  - merge
  - mergekit
  - {strategy_lower}
base_model:
{base_model_list}
---

# {repo_id}

Bu model {num_models} Türkçe LLM'in **{strategy}** yöntemiyle birleştirilmesiyle
oluşturulmuştur. Herhangi bir ek eğitim yapılmamıştır — sadece ağırlık
aritmetiği uygulanmıştır.

## Merge Detayları
- **Yöntem:** {strategy}
- **Araç:** [mergekit](https://github.com/arcee-ai/mergekit)
- **Parametreler:** {parameters}

## Kaynak Modeller

{model_table}

## Benchmark Sonuçları

{benchmark_table}

## Kullanım

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}",
                                              device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tokenizer("Türkiye'nin başkenti", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

## Metodoloji ve Kaynak Kod

📂 [github.com/CengizhanBayram/experiment_of_merging](https://github.com/CengizhanBayram/experiment_of_merging)

---

> Merged from [{source_models_str}] using {strategy} strategy.
> Benchmarks and methodology: github.com/CengizhanBayram/experiment_of_merging
"""


def get_benchmark_table(strategy_key: str) -> str:
    """benchmark_results.json'dan tablo oluşturur."""
    results_path = PROJECT_ROOT / "results" / "benchmark_results.json"

    if not results_path.exists():
        return "| Metrik | Değer |\n|--------|-------|\n| Perplexity | _Henüz hesaplanmadı_ |\n| Manuel Skor | _Henüz hesaplanmadı_ |"

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        models = data.get("models", {})
        key = strategy_key.lower()

        if key not in models or models[key].get("perplexity") is None:
            return "| Metrik | Değer |\n|--------|-------|\n| Perplexity | _Henüz hesaplanmadı_ |\n| Manuel Skor | _Henüz hesaplanmadı_ |"

        result = models[key]
        ppl = result.get("perplexity", "N/A")
        score = result.get("manual_score", "N/A")

        table = "| Metrik | Değer |\n|--------|-------|\n"
        table += f"| Türkçe Perplexity ↓ | {ppl} |\n"
        table += f"| Manuel Skor (20 soru) | {score} |"

        return table

    except Exception:
        return "| Metrik | Değer |\n|--------|-------|\n| Perplexity | _Hata_ |\n| Manuel Skor | _Hata_ |"


def create_model_table(source_models: list) -> str:
    """Kaynak modelleri markdown tablosu olarak oluşturur."""
    table = "| # | Model | HuggingFace Linki |\n|---|-------|-------------------|\n"
    for i, model in enumerate(source_models, 1):
        short_name = model.split("/")[-1]
        table += f"| {i} | {short_name} | [{model}](https://huggingface.co/{model}) |\n"
    return table


def create_model_card(
    repo_id: str,
    strategy: str,
    source_models: list,
    parameters: str,
) -> str:
    """Model card'ı oluşturur."""
    base_model_list = "\n".join(f"  - {m}" for m in source_models)
    model_table = create_model_table(source_models)
    benchmark_table = get_benchmark_table(strategy)

    card = MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        strategy=strategy,
        strategy_lower=strategy.lower(),
        num_models=len(source_models),
        parameters=parameters,
        base_model_list=base_model_list,
        model_table=model_table,
        benchmark_table=benchmark_table,
        source_models_str=", ".join(source_models),
    )

    return card


def push_model(
    model_path: str,
    repo_id: str,
    strategy: str,
    source_models: list,
    token: str,
):
    """Modeli HuggingFace Hub'a yükler."""
    print(f"\n{'='*70}")
    print(f"🚀 HUGGINGFACE'E YÜKLEME BAŞLIYOR")
    print(f"{'='*70}")
    print(f"  📂 Model   : {model_path}")
    print(f"  🏷️  Repo    : {repo_id}")
    print(f"  🔀 Strateji: {strategy}")
    print(f"  📝 Kaynak  : {', '.join(source_models)}")
    print()

    # HF Login
    try:
        login(token=token)
        print("  ✅ HuggingFace girişi başarılı.")
    except Exception as e:
        print(f"  ❌ HuggingFace giriş hatası: {e}")
        sys.exit(1)

    # Model ve tokenizer yükle
    print("  📥 Model ve tokenizer yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("  ✅ Model yüklendi.")
    except Exception as e:
        print(f"  ❌ Model yükleme hatası: {e}")
        sys.exit(1)

    # Model card oluştur
    parameters = STRATEGY_PARAMS.get(strategy.upper(), "N/A")
    model_card = create_model_card(repo_id, strategy, source_models, parameters)

    # Model card'ı kaydet
    card_path = Path(model_path) / "README.md"
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    print(f"  📝 Model card oluşturuldu: {card_path}")

    # Push
    print(f"\n  ⬆️  Model HuggingFace'e yükleniyor: {repo_id}")
    print("  ⏳ Bu işlem birkaç dakika sürebilir...")

    try:
        # Tokenizer push
        print("  📤 Tokenizer yükleniyor...")
        tokenizer.push_to_hub(
            repo_id,
            token=token,
            commit_message=f"Upload {strategy} merged tokenizer",
        )
        print("  ✅ Tokenizer yüklendi.")

        # Model push
        print("  📤 Model yükleniyor...")
        model.push_to_hub(
            repo_id,
            token=token,
            commit_message=f"Upload {strategy} merged model",
        )
        print("  ✅ Model yüklendi.")

        # Model card push
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Update model card for {strategy} merge",
        )
        print("  ✅ Model card yüklendi.")

    except Exception as e:
        print(f"  ❌ Push hatası: {e}")
        sys.exit(1)

    hf_url = f"https://huggingface.co/{repo_id}"
    print(f"\n  {'─'*60}")
    print(f"  ✅ BAŞARILI! Model yüklendi.")
    print(f"  🔗 URL: {hf_url}")
    print(f"  {'─'*60}")

    return hf_url


def main():
    parser = argparse.ArgumentParser(
        description="Merge edilmiş Türkçe LLM modellerini HuggingFace Hub'a yükler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python push_to_hub.py \\
      --model_path ./merged_models/slerp \\
      --repo_id Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1 \\
      --strategy SLERP \\
      --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0"

  python push_to_hub.py \\
      --model_path ./merged_models/ties \\
      --repo_id Cosmobillian/TR-Llama-8B-3way_TIES_v1 \\
      --strategy TIES \\
      --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0,malhajar/Mistral-7b-tr"
        """,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Merge edilmiş model dizini",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (örn: Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["SLERP", "TIES", "DARE", "slerp", "ties", "dare"],
        help="Merge stratejisi",
    )
    parser.add_argument(
        "--source_models",
        type=str,
        required=True,
        help="Kaynak modeller (virgülle ayrılmış)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (varsayılan: HF_TOKEN ortam değişkeni)",
    )
    args = parser.parse_args()

    # Token kontrolü
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HuggingFace token bulunamadı!")
        print("   Seçenekler:")
        print("   1. --token argümanı ile verin")
        print("   2. HF_TOKEN ortam değişkenini ayarlayın")
        print("   3. Colab'da: from google.colab import userdata; HF_TOKEN = userdata.get('HF_TOKEN')")
        sys.exit(1)

    # Model path kontrolü
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model dizini bulunamadı: {model_path}")
        sys.exit(1)

    # Source modelleri parse et
    source_models = [m.strip() for m in args.source_models.split(",")]
    strategy = args.strategy.upper()

    # Push
    url = push_model(
        model_path=str(model_path),
        repo_id=args.repo_id,
        strategy=strategy,
        source_models=source_models,
        token=token,
    )

    print(f"\n{'='*70}")
    print(f"✅ TAMAMLANDI: {url}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
