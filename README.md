# 𓂀 KhemetCode AI

> **Ancient Egyptian wisdom meets modern code generation.**
> A Text-to-Code Transformer model with a full-stack web interface, API key management, and incremental fine-tuning.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Text → Code** | Generate Java, Python, JS, Go, C++ code from natural language |
| 🌡 **Temperature + Top-K** | Full sampling control per request |
| 🎙 **Voice Input** | Speak your prompt via Web Speech API |
| ⌨️ **Typewriter Reveal** | Code types out character by character |
| 📊 **Code Metrics** | Lines, tokens, cyclomatic complexity on every response |
| 🏋️ **On-the-fly Training** | Fine-tune the model with new prompt/code pairs without losing previous knowledge |
| 🌐 **Language-aware Training** | Tag training pairs by language (Python, Java, JS, etc.) |
| 🔑 **API Key Manager** | Generate, revoke, delete keys — track usage per key |
| 🧪 **API Playground** | Standalone tester UI on a separate server (port 5051) |
| 🏛 **Egyptian Intro Page** | Animated pyramid, hieroglyphs, and particle effects |
| 💬 **Export Chat** | Download full conversation as Markdown |

---

## 🗂 Project Structure

```
files_for_model/
├── app.py                   # Main Flask server (port 5050)
├── playground_server.py     # Standalone playground server (port 5051)
├── khemetcode_demo.py       # Python API demo script
├── test_api_key.py          # API key test suite
├── static/
│   ├── intro.html           # Animated entry page
│   ├── index.html           # Chat UI
│   ├── train.html           # Training Studio
│   ├── keys.html            # API Key Manager
│   └── playground.html      # API Playground
├── best_transformer.keras   # ⚠️ NOT included (too large)
├── nl_tokenizer.model       # ⚠️ NOT included
└── code_tokenizer.model     # ⚠️ NOT included
```

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install flask flask-cors tensorflow sentencepiece numpy==1.26.4 ml_dtypes==0.3.2
```

### 2. Add model files
Place these in the project root (not in the repo — too large):
- `best_transformer.keras`
- `nl_tokenizer.model`
- `code_tokenizer.model`

### 3. Start the main server
```bash
python3 app.py
```
Opens at → **http://127.0.0.1:5050**

### 4. Start the playground server (optional, separate)
```bash
python3 playground_server.py
```
Opens at → **http://127.0.0.1:5051**

---

## 🌐 Pages

| URL | Page |
|---|---|
| `http://127.0.0.1:5050/` | 🏛 Intro / Splash |
| `http://127.0.0.1:5050/chat` | 💬 Chat Interface |
| `http://127.0.0.1:5050/train` | 🏋️ Training Studio |
| `http://127.0.0.1:5050/keys` | 🔑 API Key Manager |
| `http://127.0.0.1:5051/` | 🧪 API Playground |

---

## 🔑 API Usage

### Generate code via Python

```python
import requests

r = requests.post("http://127.0.0.1:5050/api/generate",
    headers={"X-API-Key": "your-key-here"},
    json={
        "text": "print hello world with python",
        "temperature": 0.8,
        "max_len": 150,
        "top_k": 0
    })

print(r.json()["code"])
print(r.json()["metrics"])
```

### REST Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/generate` | Generate code from text |
| `POST` | `/api/train` | Fine-tune with new pair |
| `GET` | `/api/status` | Model status |
| `GET` | `/api/model_info` | Architecture & training stats |
| `GET` | `/api/keys` | List API keys |
| `POST` | `/api/keys` | Create API key |
| `POST` | `/api/keys/revoke` | Revoke a key |
| `POST` | `/api/keys/delete` | Delete a key |

---

## 🏗 Model Architecture

| Param | Value |
|---|---|
| Type | Transformer (Encoder-Decoder) |
| Layers | 4 encoder + 4 decoder |
| Model dim | 256 |
| Attention heads | 8 |
| FFN dim | 1024 |
| Vocab size | 8,000 (enc + dec) |
| Parameters | ~12M |
| Base dataset | CodeXGLUE text-to-code |
| Best val accuracy | 68.65% |

---

## 🛡 License

MIT — built for research and educational purposes.

---

*𓂀 KhemetCode — where the ancient oracle speaks in code.*
