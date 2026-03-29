# 🇹🇷 Turkish LLM Model Merging

Türkçe açık kaynak LLM modellerini **SLERP**, **TIES** ve **DARE** merge stratejileriyle birleştirme, benchmark ile karşılaştırma ve HuggingFace'de paylaşma projesi.

Hiç eğitim yapılmadı. Sıfır GPU maliyeti. Sadece ağırlık aritmetiği.

---

## 🤗 HuggingFace Modeller

| Strateji | Model |
|----------|-------|
| SLERP | [Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1](https://huggingface.co/Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1) |
| TIES  | [Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_TIES_v1](https://huggingface.co/Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_TIES_v1) |
| DARE  | [Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_DARE_v1](https://huggingface.co/Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_DARE_v1) |

---

## 📊 Benchmark Sonuçları

| Model | Strateji | Perplexity ↓ | Manuel /20 |
|-------|----------|:------------:|:----------:|
| TR-Llama-8B-Cosmos-Trendyol_SLERP_v1 | SLERP | **39.31** ✅ | 20/20 |
| TR-Llama-8B-Cosmos-Trendyol_TIES_v1  | TIES  | 45.18 | 18/20 |
| TR-Llama-8B-Cosmos-Trendyol_DARE_v1  | DARE  | 72.10 ❌ | 14/20 |
| Baseline — Turkish-Llama-8B (YTÜ Cosmos) | — | 57.35 | 18/20 |
| Baseline — Trendyol-LLM-8B | — | 39.87 | 19/20 |

**SLERP her iki kaynak modeli de geride bıraktı.**
DARE ise density=0.5 ile çok agresif davrandı — Türkçe anlama kapasitesi belirgin biçimde düştü.

> Manuel skor 20 sorudan oluşmaktadır: 5 genel bilgi, 5 matematik, 5 gramer, 5 instruction following. Keyword-matching ile hesaplanmıştır.

---

## 🧠 Kullanılan Modeller

| Model | Açıklama |
|-------|----------|
| `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` | Türkçe instruction-tuned Llama (Model A) |
| `Trendyol/Trendyol-LLM-8b-chat-v2.0` | Trendyol Türkçe chat modeli (Model B) |

> Üçüncü bir model (Mistral-7B-tr) planlanmıştı ancak tokenizer uyumsuzluğu nedeniyle merge dışında kaldı.

---

## 🔀 Merge Stratejileri

### SLERP — Spherical Linear Interpolation
İki modeli küresel interpolasyonla harmanlıyor. `t=0.5` ile eşit ağırlık.
En temiz ve tutarlı yöntem.

### TIES — TrIm, Elect, Sum
Çakışan ağırlıkları "trim" ederek gürültüyü azaltıyor.
`weights=[0.5, 0.3, 0.2]`, `density=0.7`

### DARE — Drop And REscale
Ağırlıkların bir kısmını silerek daha seyrek bir model elde ediyor.
`weights=[0.4, 0.35, 0.25]`, `density=0.5`

---

## 🚀 Hızlı Başlangıç

### Google Colab (Önerilen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CengizhanBayram/experiment_of_merging/blob/main/colab_full_pipeline.ipynb)

`colab_full_pipeline.ipynb` dosyasını açın ve **Run All** yapın. T4 GPU yeterlidir.

### Yerel Kurulum

```bash
git clone https://github.com/CengizhanBayram/experiment_of_merging.git
cd experiment_of_merging
pip install -r requirements.txt
```

---

## 📁 Dosya Yapısı

```
experiment_of_merging/
├── README.md
├── requirements.txt
├── configs/
│   ├── slerp_config.yaml
│   ├── ties_config.yaml
│   └── dare_config.yaml
├── scripts/
│   ├── check_tokenizers.py        # Tokenizer uyumluluk kontrolü
│   ├── run_merge.py               # Merge çalıştırıcı
│   ├── benchmark.py               # Benchmark değerlendirme
│   └── push_to_hub.py             # HuggingFace'e yükleme
├── results/
│   └── benchmark_results.json
└── colab_full_pipeline.ipynb
```

---

## 🛠️ Kullanım

```bash
# Tokenizer kontrolü
python scripts/check_tokenizers.py

# Merge
python scripts/run_merge.py --strategy slerp
python scripts/run_merge.py --strategy ties
python scripts/run_merge.py --strategy dare

# Benchmark (tüm modeller)
python scripts/benchmark.py --model all

# HuggingFace'e push
python scripts/push_to_hub.py \
    --model_path ./merged_models/slerp \
    --repo_id Cosmobillian/TR-Llama-8B-Cosmos-Trendyol_SLERP_v1 \
    --strategy SLERP \
    --source_models "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1,Trendyol/Trendyol-LLM-8b-chat-v2.0"
```

`HF_TOKEN` ortam değişkeni gereklidir.

---

## ⚙️ Gereksinimler

- Python 3.10+
- CUDA destekli GPU (Colab T4 yeterli)
- ~40 GB disk alanı
- HuggingFace hesabı ve write token

---

## 📬 İletişim

- **LinkedIn:** [Cengizhan Bayram](https://www.linkedin.com/in/cengizhan-bayram)
- **HuggingFace:** [Cosmobillian](https://huggingface.co/Cosmobillian)
- **GitHub:** [CengizhanBayram](https://github.com/CengizhanBayram)

---

## 📜 Lisans

MIT License. Merge edilen modeller kendi lisanslarına tabidir (Llama 3 License).
