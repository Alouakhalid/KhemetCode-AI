<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=800&size=32&pause=1000&color=C9A227&center=true&vCenter=true&width=600&lines=%F0%93%82%80+KhemetCode+AI;Egyptian+Oracle+of+Code;Text+%E2%86%92+Code+Transformer" alt="KhemetCode AI" />

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Transformer-C9A227?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-00C9A7?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Params-~12M-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/Val%20Accuracy-68.65%25-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-CodeXGLUE-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Languages-Java%20%7C%20Python%20%7C%20JS%20%7C%20Go%20%7C%20C%2B%2B-informational?style=flat-square"/>
  <img src="https://img.shields.io/github/stars/Alouakhalid/KhemetCode-AI?style=flat-square&color=yellow"/>
</p>

<br/>

> **𓂀 Ancient Egyptian wisdom meets modern code generation.**  
> A custom-built Encoder-Decoder Transformer that converts natural language descriptions into code,  
> wrapped in a production-ready web interface with real-time fine-tuning, API key management, and voice input.

<br/>

```
𓂀  "Describe it in words — the Oracle will speak it in code."  𓂀
```

</div>

---

## 📖 Table of Contents

- [✨ Features](#-features)
- [🏗 Architecture](#-architecture)
- [🖥 Interface Pages](#-interface-pages)
- [🚀 Quick Start](#-quick-start)
- [🔑 API Reference](#-api-reference)
- [🧪 API Playground](#-api-playground)
- [🏋️ Training Studio](#️-training-studio)
- [📊 Model Performance](#-model-performance)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#️-configuration)
- [🛡 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔮 Core AI
- **Text → Code Transformer** — Encoder-Decoder architecture trained on CodeXGLUE
- **Temperature Control** — Scale from deterministic (0) to creative (2.0)
- **Top-K Sampling** — Restrict vocabulary for coherent generation
- **Auto Language Detection** — Python, Java, JS, TS, Go, C++ auto-labelled
- **Code Metrics** — Lines, tokens, cyclomatic complexity per response

</td>
<td width="50%">

### 🌐 Web Interface
- **ChatGPT-style UI** — Dark Egyptian-themed chat interface
- **Animated Intro Page** — Pyramid, hieroglyphs, particle starfield
- **Typewriter Code Reveal** — Code types out with blinking cursor
- **🎙 Voice Input** — Speak prompts via Web Speech API (Chrome)
- **⬇ Export Chat** — Download conversation as Markdown

</td>
</tr>
<tr>
<td width="50%">

### 🏋️ Training
- **On-the-fly Fine-tuning** — Train on new pairs without catastrophic forgetting
- **Language-aware Pairs** — Tag training data by language
- **Cumulative Learning** — All pairs persisted in `learned_data.json`
- **Background Training** — Non-blocking thread with live log streaming
- **Session History** — Every training session timestamped and tracked

</td>
<td width="50%">

### 🔑 API & Developer Tools
- **API Key Manager** — Generate, revoke, delete UUID keys
- **Usage Tracking** — Per-key call count and last-used timestamp
- **REST API** — Full JSON API with `X-API-Key` header auth
- **Standalone Playground** — Separate server (port 5051) for API testing
- **Python Demo** — Ready-to-use `khemetcode_demo.py` script

</td>
</tr>
</table>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KhemetCode Transformer                    │
│                                                             │
│   Natural Language Input                                    │
│         │                                                   │
│         ▼                                                   │
│   ┌──────────────┐        ┌──────────────────────────┐     │
│   │   Encoder    │        │        Decoder           │     │
│   │  4 Layers    │──────▶│       4 Layers            │     │
│   │  256 d_model │        │  Masked Self-Attention    │     │
│   │  8 Heads     │        │  Cross-Attention          │     │
│   │  1024 FFN    │        │  Feed-Forward             │     │
│   │  GELU act.   │        │  LayerNorm (Pre-LN)      │     │
│   └──────────────┘        └──────────────────────────┘     │
│         │                              │                    │
│   BPE Tokenizer               Temperature + Top-K          │
│   (SentencePiece)             Softmax Sampling             │
│   8,000 vocab                         │                    │
│                                       ▼                    │
│                              Generated Code Tokens         │
│                              BPE Detokenization            │
│                                       │                    │
│                                  Code Output               │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Details

| Component | Specification |
|---|---|
| Architecture | Transformer Encoder-Decoder |
| Total Parameters | ~12 Million |
| Encoder Layers | 4 |
| Decoder Layers | 4 |
| Model Dimension | 256 |
| Attention Heads | 8 |
| FFN Dimension | 1024 |
| Activation | GELU |
| Normalization | Pre-LayerNorm (ε=1e-6) |
| Dropout | 0.1 |
| Tokenizer | SentencePiece BPE |
| Vocabulary | 8,000 (encoder + decoder) |
| Max Encoder Length | 50 tokens |
| Max Decoder Length | 200 tokens |
| Special Tokens | `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3` |

---

## 🖥 Interface Pages

### 🏛 Intro Page — `http://127.0.0.1:5050/`
Animated Egyptian-themed entry with:
- Animated SVG pyramid with Eye of Horus
- Rotating hieroglyph ring
- Star particle network canvas
- Code-rain animation (Matrix-style with Egyptian glyphs)
- Model loading progress bar
- Particle burst on entry click

### 💬 Chat UI — `http://127.0.0.1:5050/chat`
- Real-time code generation with typewriter animation
- Temperature + Top-K sliders
- Voice input microphone button
- Code metrics badge (Lines · Tokens · Complexity)
- Model Info modal with architecture stats
- Export conversation as Markdown
- Session history sidebar

### 🏋️ Training Studio — `http://127.0.0.1:5050/train`
- Language selector (Java, Python, JS, TS, C++, Go)
- Epoch and learning rate controls
- Live training log stream
- Learned pairs history with language badges

### 🔑 API Keys — `http://127.0.0.1:5050/keys`
- Generate UUID-based API keys
- One-time key reveal with copy banner
- Usage bars and last-used timestamps
- Revoke / Delete with confirmation

### 🧪 Playground — `http://127.0.0.1:5051/`
- Completely standalone (separate server)
- API key validation with Verify button
- Full parameter controls
- Session history (last 10 generations)
- Raw JSON response viewer

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
pip
```

### 1. Clone the repository

```bash
git clone https://github.com/Alouakhalid/KhemetCode-AI.git
cd KhemetCode-AI
```

### 2. Install dependencies

```bash
pip install flask flask-cors tensorflow sentencepiece numpy==1.26.4 ml_dtypes==0.3.2
```

### 3. Add model files ⚠️

The following files are **not included** in the repo (too large). Place them in the project root:

```
KhemetCode-AI/
├── best_transformer.keras    ← Required (model weights)
├── nl_tokenizer.model        ← Required (encoder tokenizer)
└── code_tokenizer.model      ← Required (decoder tokenizer)
```

### 4. Start the main server

```bash
python3 app.py
```

Visit → **http://127.0.0.1:5050**

### 5. Start the playground (optional)

```bash
python3 playground_server.py
```

Visit → **http://127.0.0.1:5051**

---

## 🔑 API Reference

### Authentication

All API requests optionally accept an API key:

```http
X-API-Key: your-key-here
```

Or in the request body:

```json
{ "api_key": "your-key-here" }
```

> Keys can be generated at `http://127.0.0.1:5050/keys`

---

### `POST /api/generate`

Generate code from a natural language prompt.

**Request**
```json
{
  "text":        "check if a list is sorted in ascending order",
  "temperature": 0.8,
  "max_len":     150,
  "top_k":       0
}
```

**Response**
```json
{
  "code":        "boolean isSorted(int[] arr) { ... }",
  "temperature": 0.8,
  "top_k":       0,
  "input":       "check if a list is sorted in ascending order",
  "metrics": {
    "lines":      4,
    "tokens":     18,
    "max_depth":  1,
    "complexity": 2
  },
  "timestamp": "2026-05-17T04:29:36.370403"
}
```

| Parameter | Type | Default | Range | Description |
|---|---|---|---|---|
| `text` | string | required | — | Natural language prompt |
| `temperature` | float | 1.0 | 0.0 – 2.0 | Sampling temperature |
| `max_len` | int | 200 | 10 – 200 | Max output tokens |
| `top_k` | int | 0 | 0 – 200 | Top-K filtering (0 = off) |

---

### `POST /api/train`

Fine-tune the model with a new prompt/code pair.

**Request**
```json
{
  "nl_text":    "print hello world",
  "code_text":  "System.out.println(\"Hello World\");",
  "lang":       "java",
  "epochs":     3,
  "learn_rate": 0.0001
}
```

**Response**
```json
{ "status": "Training started", "pairs": 5 }
```

---

### `GET /api/status`

```json
{
  "model_ready":   true,
  "is_training":   false,
  "model_name":    "KhemetCode",
  "version":       "1.0",
  "learned_pairs": 5
}
```

---

### `GET /api/model_info`

Returns full architecture details, training history, and live stats.

---

### API Key Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `GET` | `/api/keys` | — | List all keys (masked) |
| `POST` | `/api/keys` | `{"name": "My App"}` | Create new key |
| `POST` | `/api/keys/revoke` | `{"key": "uuid"}` | Revoke a key |
| `POST` | `/api/keys/delete` | `{"key": "uuid"}` | Delete a key |

---

## 🧪 API Playground

The playground is a **completely separate web server** (port 5051) that lets anyone test the API with their key — no access to training or admin pages.

```bash
python3 playground_server.py
```

Features:
- Paste your API key → click **Verify** to validate instantly
- Write any prompt → `Ctrl+Enter` or click Generate
- See syntax-highlighted output with auto language detection
- Session history of last 10 generations
- Raw JSON response viewer

### Quick test with Python

```python
import requests

API_KEY = "your-key-from-/keys"

r = requests.post("http://127.0.0.1:5050/api/generate",
    headers={"X-API-Key": API_KEY},
    json={
        "text":        "print hello world with python",
        "temperature": 0.8,
        "max_len":     100
    })

d = r.json()
print(d["code"])
# → print('hello world')

print(d["metrics"])
# → {'lines': 1, 'tokens': 2, 'max_depth': 0, 'complexity': 1}
```

---

## 🏋️ Training Studio

The model can be fine-tuned on new examples **without losing previous knowledge** — using cumulative replay training.

### How it works

```
1. User provides (prompt, code, language) pair
2. Pair saved to learned_data.json
3. Background thread fine-tunes on ALL accumulated pairs
4. Model saved back to best_transformer.keras
5. Next generation immediately uses updated weights
```

### Supported Training Languages

| Language | Tag |
|---|---|
| ☕ Java | `java` |
| 🐍 Python | `python` |
| 🟨 JavaScript | `javascript` |
| 🔷 TypeScript | `typescript` |
| ⚙️ C++ | `cpp` |
| 🐹 Go | `go` |

### Fine-tuning Parameters

| Parameter | Range | Recommended |
|---|---|---|
| Epochs | 1 – 20 | 3 – 5 |
| Learning Rate | 1e-6 – 1e-2 | 1e-4 |
| Batch Size | auto (all pairs) | — |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Training Epochs | 11 |
| Best Validation Loss | 1.6575 |
| Best Validation Accuracy | **68.65%** |
| Base Dataset | CodeXGLUE text-to-code |
| Dataset Size | ~100,000 pairs |
| Training Framework | TensorFlow / Keras |

### Sampling Modes

| Mode | Temperature | Top-K | Behaviour |
|---|---|---|---|
| Greedy | 0.0 | 0 | Most likely token always |
| Balanced | 0.8 | 0 | Good quality + some variety |
| Creative | 1.2 | 20 | More diverse outputs |
| Experimental | 1.8 | 50 | Highly varied, may be noisy |

---

## 📁 Project Structure

```
KhemetCode-AI/
│
├── app.py                        # Main Flask server (port 5050)
├── playground_server.py          # Standalone playground server (port 5051)
│
├── static/
│   ├── intro.html                # 🏛 Animated Egyptian intro page
│   ├── index.html                # 💬 Chat interface
│   ├── train.html                # 🏋️ Training Studio
│   ├── keys.html                 # 🔑 API Key Manager
│   └── playground.html          # 🧪 API Playground UI
│
├── khemetcode_demo.py            # Python API demo script
├── test_api_key.py               # API key test suite
├── text_to_code_transformer.ipynb # Original training notebook
│
├── code_tokenizer.vocab          # Decoder vocabulary
├── nl_tokenizer.vocab            # Encoder vocabulary
│
├── learned_data.json             # ⚙️ Auto-generated (fine-tune pairs)
├── api_keys.json                 # ⚙️ Auto-generated (API keys store)
│
├── best_transformer.keras        # ❌ NOT in repo — add manually
├── nl_tokenizer.model            # ❌ NOT in repo — add manually
├── code_tokenizer.model          # ❌ NOT in repo — add manually
│
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

Key constants in `app.py`:

```python
MAX_ENC_LEN    = 50      # Max encoder input tokens
MAX_DEC_LEN    = 200     # Max decoder output tokens
ENC_VOCAB_SIZE = 8000    # Encoder vocabulary size
DEC_VOCAB_SIZE = 8000    # Decoder vocabulary size
EMBEDDING_DIM  = 256     # Model dimension
```

Server ports:

```python
# app.py — main server
app.run(host="0.0.0.0", port=5050)

# playground_server.py — playground
app.run(host="0.0.0.0", port=5051)
```

---

## 🛡 License

```
MIT License — free to use, modify, and distribute.
```

---

<div align="center">

**Built with 𓂀 by [Alouakhalid](https://github.com/Alouakhalid)**

*Where the ancient oracle speaks in code.*

```
𓂀  𓆣  𓅓  𓃭  𓆏  𓀭  𓁹
```

<img src="https://img.shields.io/badge/Made%20with-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Served%20with-Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Styled%20with-Vanilla%20CSS-1572B6?style=for-the-badge&logo=css3&logoColor=white"/>

</div>
