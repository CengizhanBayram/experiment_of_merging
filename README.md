# 🇹🇷 Turkish LLM Model Merging

Türkçe açık kaynak LLM modellerini **SLERP**, **TIES** ve **DARE** merge stratejileriyle birleştirme, benchmark ile karşılaştırma ve HuggingFace'de paylaşma projesi.

## 📋 Proje Özeti

| Strateji | Çıktı Model | Kaynak Modeller |
|----------|-------------|-----------------|
| **SLERP** | [Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1](https://huggingface.co/Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1) | Model A + Model B |
| **TIES** | [Cosmobillian/TR-Llama-8B-3way_TIES_v1](https://huggingface.co/Cosmobillian/TR-Llama-8B-3way_TIES_v1) | Model A + B + C |
| **DARE** | [Cosmobillian/TR-Llama-8B-3way_DARE_v1](https://huggingface.co/Cosmobillian/TR-Llama-8B-3way_DARE_v1) | Model A + B + C |

### Kullanılan Modeller

| Kısaltma | Model | Açıklama |
|----------|-------|----------|
| Model A (primary) | `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | Türkçe instruction-tuned Llama |
| Model B (secondary) | `Trendyol/Trendyol-LLM-8b-chat-v2.0` | Trendyol Türkçe chat modeli |
| Model C (tertiary) | `malhajar/Mistral-7b-tr` | Türkçe Mistral (sadece TIES/DARE) |
| Base | `meta-llama/Meta-Llama-3-8B` | Base model (TIES/DARE task vector hesabı için) |

## 🚀 Hızlı Başlangıç

### Google Colab (Önerilen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cosmobillian/turkish-llm-merging/blob/main/colab_full_pipeline.ipynb)

`colab_full_pipeline.ipynb` dosyasını açın ve **"Run All"** yapın. T4 GPU yeterlidir.

### Yerel Kurulum

```bash
git clone https://github.com/Cosmobillian/turkish-llm-merging.git
cd turkish-llm-merging
pip install -r requirements.txt
```

## 📁 Dosya Yapısı

```
turkish-llm-merging/
├── README.md                      # Bu dosya
├── requirements.txt               # Python bağımlılıkları
├── configs/
│   ├── slerp_config.yaml          # SLERP merge yapılandırması
│   ├── ties_config.yaml           # TIES merge yapılandırması
│   └── dare_config.yaml           # DARE merge yapılandırması
├── scripts/
│   ├── check_tokenizers.py        # Tokenizer uyumluluk kontrolü
│   ├── run_merge.py               # Merge çalıştırıcı
│   ├── benchmark.py               # Benchmark değerlendirme
│   └── push_to_hub.py             # HuggingFace'e yükleme
├── results/
│   └── benchmark_results.json     # Benchmark sonuçları
└── colab_full_pipeline.ipynb      # Tek notebook, tüm adımları içerir
```

## 🛠️ Kullanım

### 1. Tokenizer Kontrolü

```bash
python scripts/check_tokenizers.py
```

Üç modelin tokenizer'larını karşılaştırır: vocab_size, special_tokens, padding_side. Uyumsuz çiftleri tespit eder ve hangi modelin çıkarılması gerektiğini önerir.

### 2. Merge Çalıştırma

```bash
# SLERP merge (2 model)
python scripts/run_merge.py --strategy slerp

# TIES merge (3 model)
python scripts/run_merge.py --strategy ties

# DARE merge (3 model)
python scripts/run_merge.py --strategy dare

# Özel config ile
python scripts/run_merge.py --strategy slerp --config ./my_config.yaml

# Özel çıktı dizini
python scripts/run_merge.py --strategy ties --output ./my_model
```

Her merge sonrası 3 Türkçe test cümlesi üretilerek sanity check yapılır.

### 3. Benchmark

```bash
# Tüm modelleri değerlendir
python scripts/benchmark.py --model all

# Tek model
python scripts/benchmark.py --model ./merged_models/slerp

# Özel çıktı dosyası
python scripts/benchmark.py --model all --output results/my_results.json
```

**Ölçülen metrikler:**
- **Türkçe Perplexity** — mc4/tr veri seti üzerinde (ilk 500 örnek)
- **Manuel Skor (20 soru):**
  - 5 genel bilgi (Türkiye tarihi, coğrafya)
  - 5 matematik
  - 5 gramer / Türk dili
  - 5 instruction following

### 4. HuggingFace'e Yükleme

```bash
# SLERP modelini yükle
python scripts/push_to_hub.py \
    --model_path ./merged_models/slerp \
    --repo_id Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1 \
    --strategy SLERP \
    --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0"

# TIES modelini yükle
python scripts/push_to_hub.py \
    --model_path ./merged_models/ties \
    --repo_id Cosmobillian/TR-Llama-8B-3way_TIES_v1 \
    --strategy TIES \
    --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0,malhajar/Mistral-7b-tr"

# DARE modelini yükle
python scripts/push_to_hub.py \
    --model_path ./merged_models/dare \
    --repo_id Cosmobillian/TR-Llama-8B-3way_DARE_v1 \
    --strategy DARE \
    --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0,malhajar/Mistral-7b-tr"
```

`HF_TOKEN` ortam değişkeni gereklidir.

## 🔀 Merge Stratejileri

### SLERP (Spherical Linear Interpolation)
- **2 model** arasında küresel doğrusal interpolasyon
- `t=0.5` → eşit ağırlık (alternatifler: 0.3, 0.7)
- En basit ve tutarlı yöntem

### TIES (TrIm, Elect, Sum)
- **3 model** birleştirme, base model üzerinden task vector hesabı
- Gereksiz parametreleri budayarak (trim) temiz birleştirme
- `weights=[0.5, 0.3, 0.2]`, `density=0.7`

### DARE (Drop And REscale)
- **3 model** birleştirme, TIES sign election ile birlikte
- Rastgele parametreleri düşürüp (drop) yeniden ölçekleme (rescale)
- `weights=[0.4, 0.35, 0.25]`, `density=0.5`

## 📊 Benchmark Sonuçları

Sonuçlar `results/benchmark_results.json` dosyasında saklanır.

| Model | Strateji | Perplexity ↓ | Manuel/20 | En İyi? |
|-------|----------|:------------:|:---------:|:-------:|
| SLERP | SLERP | _tbd_ | _tbd_ | |
| TIES | TIES | _tbd_ | _tbd_ | |
| DARE | DARE | _tbd_ | _tbd_ | |
| Baseline | — | _tbd_ | _tbd_ | |

> Benchmark'ları çalıştırdıktan sonra bu tablo güncellenecektir.

## ⚙️ Gereksinimler

- Python 3.10+
- CUDA destekli GPU (Colab T4 yeterli)
- ~40 GB disk alanı (model indirmeleri için)
- HuggingFace hesabı ve token (push için)

## 📜 Lisans

Bu proje MIT lisansı ile lisanslanmıştır. Merge edilen modeller kendi lisanslarına tabidir (Llama 3 License).

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📬 İletişim

- **Linkedln:** [Cengizhan Bayram](https://www.linkedin.com/in/cengizhan-bayram)
- **HuggingFace:** [Cosmobillian](https://huggingface.co/Cosmobillian)

---

> Merged from Turkish open-source LLMs using SLERP, TIES, and DARE strategies.
> Benchmarks and methodology: [github.com/Cosmobillian/turkish-llm-merging](https://github.com/Cosmobillian/turkish-llm-merging)
