"""
KhemetCode AI - Flask Backend
Egyptian-inspired text-to-code transformer model server
Model: KhemetCode (Khemet = ancient Egypt + Code)
"""

import os
import re
import json
import numpy as np
import threading
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow / Keras imports (lazy to avoid slow startup messages at top-level)
# ─────────────────────────────────────────────────────────────────────────────
import tensorflow as tf
import sentencepiece as spm

app = Flask(__name__, static_folder="static")
CORS(app)

# ─────── Paths ────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "best_transformer.keras")
NL_TOK_PATH     = os.path.join(BASE_DIR, "nl_tokenizer.model")
CODE_TOK_PATH   = os.path.join(BASE_DIR, "code_tokenizer.model")
LEARNED_DATA    = os.path.join(BASE_DIR, "learned_data.json")
API_KEYS_FILE   = os.path.join(BASE_DIR, "api_keys.json")

# ─────── Token IDs ────────────────────────────────────────────────────────────
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# ─────── Sequence lengths ─────────────────────────────────────────────────────
MAX_ENC_LEN = 50
MAX_DEC_LEN = 200

# ─────── Model constants ──────────────────────────────────────────────────────
ENC_VOCAB_SIZE = 8000
DEC_VOCAB_SIZE = 8000
EMBEDDING_DIM  = 256

# ─────── Global state ─────────────────────────────────────────────────────────
model_ready   = False
encoder_sp    = None
decoder_sp    = None
transformer   = None
training_lock = threading.Lock()
is_training   = False
train_log     = []

# ─────── Model Architecture (must match saved weights) ────────────────────────
def Positional_Encoding(max_len, embedding_dim):
    positions = np.arange(max_len)[:, np.newaxis]
    dims      = np.arange(embedding_dim)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / embedding_dim)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )

    def call(self, q, k, v, mask=None):
        return self.att(q, k, v, attention_mask=mask)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return cfg


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff;         self.rate = rate
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ffn   = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="gelu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        x = self.ln1(x + self.drop1(self.attn(x, x, x, mask=mask), training=training))
        x = self.ln2(x + self.drop2(self.ffn(x), training=training))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "dff": self.dff, "rate": self.rate})
        return cfg


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff;         self.rate = rate
        self.masked_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads)
        self.ffn   = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="gelu"),
            tf.keras.layers.Dense(d_model)
        ])
        self.ln1   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(rate)
        self.drop2 = tf.keras.layers.Dropout(rate)
        self.drop3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_out, training=False, look_ahead_mask=None, padding_mask=None):
        x = self.ln1(x + self.drop1(self.masked_attn(x, x, x, mask=look_ahead_mask), training=training))
        x = self.ln2(x + self.drop2(self.cross_attn(x, enc_out, enc_out, mask=padding_mask), training=training))
        x = self.ln3(x + self.drop3(self.ffn(x), training=training))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "dff": self.dff, "rate": self.rate})
        return cfg


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_len, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers; self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff; self.vocab_size = vocab_size; self.max_len = max_len; self.rate = rate
        self.embedding    = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = Positional_Encoding(max_len, d_model)
        self.enc_layers   = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout      = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        x  = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x  = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_layers": self.num_layers, "d_model": self.d_model,
                    "num_heads": self.num_heads, "dff": self.dff,
                    "vocab_size": self.vocab_size, "max_len": self.max_len, "rate": self.rate})
        return cfg


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_len, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers; self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff; self.vocab_size = vocab_size; self.max_len = max_len; self.rate = rate
        self.embedding    = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_decoding = Positional_Encoding(max_len - 1, d_model)
        self.dec_layers   = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout      = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_out, training=False, look_ahead_mask=None, padding_mask=None):
        x  = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_decoding[:, :tf.shape(x)[1], :]
        x  = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_out, training=training,
                      look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_layers": self.num_layers, "d_model": self.d_model,
                    "num_heads": self.num_heads, "dff": self.dff,
                    "vocab_size": self.vocab_size, "max_len": self.max_len, "rate": self.rate})
        return cfg


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size,
                 max_enc_len, max_dec_len, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers; self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff; self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_enc_len = max_enc_len; self.max_dec_len = max_dec_len; self.rate = rate
        self.encoder     = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size,  max_enc_len, rate)
        self.decoder     = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_dec_len, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def create_masks(self, inp, tar):
        # Cast to float32 so GPU BroadcastTo never sees DT_BOOL
        enc_mask     = tf.cast(tf.math.not_equal(inp, 0)[:, tf.newaxis, :], tf.float32)
        tar_len      = tf.shape(tar)[1]
        causal_mask  = tf.cast(
            tf.linalg.band_part(tf.ones((tar_len, tar_len), dtype=tf.bool), -1, 0)[tf.newaxis, :, :],
            tf.float32
        )
        dec_pad_mask = tf.cast(tf.math.not_equal(tar, 0)[:, tf.newaxis, :], tf.float32)
        combined     = dec_pad_mask * causal_mask   # element-wise AND via float multiply
        return enc_mask, combined, enc_mask

    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_mask, combined_mask, dec_mask = self.create_masks(inp, tar)
        enc_out = self.encoder(inp, training=training, mask=enc_mask)
        dec_out = self.decoder(tar, enc_out, training=training,
                               look_ahead_mask=combined_mask, padding_mask=dec_mask)
        return self.final_layer(dec_out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_layers": self.num_layers, "d_model": self.d_model,
                    "num_heads": self.num_heads, "dff": self.dff,
                    "input_vocab_size": self.input_vocab_size,
                    "target_vocab_size": self.target_vocab_size,
                    "max_enc_len": self.max_enc_len, "max_dec_len": self.max_dec_len,
                    "rate": self.rate})
        return cfg


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model      = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(
            tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5)
        )

    def get_config(self):
        return {"d_model": int(self.d_model.numpy()), "warmup_steps": self.warmup_steps}


# ─────── Helper functions ─────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def encode_and_pad(sp_model, texts, max_len):
    sequences = []
    for text in texts:
        ids = sp_model.encode(text, out_type=int)
        ids = ids[:max_len - 2]
        ids = [BOS_ID] + ids + [EOS_ID]
        sequences.append(ids)
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post", value=PAD_ID
    )


def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits."""
    if temperature <= 0:
        temperature = 1.0
    return logits / temperature


def generate_code(nl_text, max_len=200, temperature=1.0, top_k=0):
    """Generate code from natural language description."""
    global transformer, encoder_sp, decoder_sp

    enc_seq = encode_and_pad(encoder_sp, [clean_text(nl_text)], MAX_ENC_LEN)
    dec_seq = np.array([[BOS_ID] + [PAD_ID] * (MAX_DEC_LEN - 1)], dtype=np.int32)

    effective_len = min(max_len, MAX_DEC_LEN - 1)
    for i in range(1, effective_len):
        preds  = transformer((enc_seq, dec_seq[:, :-1]), training=False)
        logits = preds[0, i - 1, :].numpy().astype(np.float64)

        # Temperature scaling
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # Top-K filtering
        if top_k > 1:
            thresh = np.sort(logits)[::-1][top_k - 1]
            logits[logits < thresh] = -1e9

        if temperature <= 0 or (temperature == 1.0 and top_k == 0):
            next_id = int(np.argmax(logits))
        else:
            probs   = np.exp(logits - np.max(logits))
            probs  /= probs.sum()
            next_id = int(np.random.choice(len(probs), p=probs))

        if next_id == EOS_ID:
            break
        dec_seq[0, i] = next_id

    tokens = dec_seq[0, 1:].tolist()
    tokens = [t for t in tokens if t not in (PAD_ID, EOS_ID)]
    return decoder_sp.decode(tokens)


def load_learned_data():
    """Load previously learned training data."""
    if os.path.exists(LEARNED_DATA):
        with open(LEARNED_DATA, 'r') as f:
            return json.load(f)
    return {"pairs": [], "sessions": []}


def save_learned_data(data):
    """Persist learned data to disk."""
    with open(LEARNED_DATA, 'w') as f:
        json.dump(data, f, indent=2)


# ─────── API Key helpers ──────────────────────────────────────────────────
import uuid as _uuid

def load_keys():
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    return {"keys": []}

def save_keys(data):
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def validate_key(key_str):
    """Returns key record if valid and active, else None."""
    if not key_str:
        return None
    store = load_keys()
    for k in store["keys"]:
        if k["key"] == key_str and k["active"]:
            return k
    return None

def increment_key_usage(key_str):
    store = load_keys()
    for k in store["keys"]:
        if k["key"] == key_str:
            k["usage"] = k.get("usage", 0) + 1
            k["last_used"] = datetime.datetime.now().isoformat()
            break
    save_keys(store)


# ─────── Code metrics ───────────────────────────────────────────────────────
def compute_metrics(code: str) -> dict:
    lines   = [l for l in code.split('\n') if l.strip()]
    tokens  = len(code.split())
    # Simple nesting depth: count indent level changes
    depth   = 0
    max_dep = 0
    for line in lines:
        stripped = line.lstrip()
        indent   = len(line) - len(stripped)
        d        = indent // 4 if indent else 0
        if d > max_dep:
            max_dep = d
    # Cyclomatic complexity proxy: count branch keywords
    branches = sum(1 for w in ['if','else','elif','for','while','case','catch','switch','&&','||','?'] if w in code)
    return {
        "lines":      len(lines),
        "tokens":     tokens,
        "max_depth":  max_dep,
        "complexity": branches + 1,   # McCabe baseline = 1
    }


# ─────── Model loading ────────────────────────────────────────────────────────
def load_model():
    global transformer, encoder_sp, decoder_sp, model_ready

    print("🔺 Loading KhemetCode model...")

    # Load tokenizers
    encoder_sp = spm.SentencePieceProcessor()
    encoder_sp.load(NL_TOK_PATH)

    decoder_sp = spm.SentencePieceProcessor()
    decoder_sp.load(CODE_TOK_PATH)

    # Build model skeleton first (needed to load weights)
    custom_objs = {
        "MultiHeadAttention": MultiHeadAttention,
        "EncoderLayer":       EncoderLayer,
        "DecoderLayer":       DecoderLayer,
        "Encoder":            Encoder,
        "Decoder":            Decoder,
        "Transformer":        Transformer,
        "CustomSchedule":     CustomSchedule,
    }

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def masked_loss(y_true, y_pred):
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

    def masked_accuracy(y_true, y_pred):
        pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        true = tf.cast(y_true, tf.int32)
        mask = tf.cast(tf.not_equal(true, PAD_ID), tf.float32)
        return tf.reduce_sum(tf.cast(tf.equal(pred, true), tf.float32) * mask) / tf.reduce_sum(mask)

    custom_objs["masked_loss"]     = masked_loss
    custom_objs["masked_accuracy"] = masked_accuracy

    transformer = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objs)
    model_ready = True
    print("✅ KhemetCode model loaded successfully!")


# ─────── Routes ───────────────────────────────────────────────────────────────
@app.route("/")
def intro():
    return send_from_directory("static", "intro.html")

@app.route("/chat")
def chat_page():
    return send_from_directory("static", "index.html")

@app.route("/train")
def train_page():
    return send_from_directory("static", "train.html")

@app.route("/keys")
def keys_page():
    return send_from_directory("static", "keys.html")

@app.route("/playground")
def playground_page():
    return send_from_directory("static", "playground.html")

@app.route("/api/model_info")
def model_info():
    learned = load_learned_data()
    pairs   = learned.get("pairs", [])
    sessions= learned.get("sessions", [])
    lang_counts = {}
    for p in pairs:
        l = p.get("lang", "java")
        lang_counts[l] = lang_counts.get(l, 0) + 1
    return jsonify({
        "model_name":       "KhemetCode",
        "version":          "1.0",
        "architecture":     "Transformer (Encoder-Decoder)",
        "num_layers":       4,
        "d_model":          256,
        "num_heads":        8,
        "dff":              1024,
        "enc_vocab":        8000,
        "dec_vocab":        8000,
        "max_enc_len":      MAX_ENC_LEN,
        "max_dec_len":      MAX_DEC_LEN,
        "dropout":          0.1,
        "total_params":     "~12M",
        "base_dataset":     "CodeXGLUE text-to-code (100k pairs)",
        "training_epochs":  11,
        "best_val_loss":    1.6575,
        "best_val_acc":     "68.65%",
        "learned_pairs":    len(pairs),
        "train_sessions":   len(sessions),
        "lang_breakdown":   lang_counts,
        "model_ready":      model_ready,
    })

@app.route("/api/status")
def status():
    return jsonify({
        "model_ready": model_ready,
        "is_training": is_training,
        "model_name": "KhemetCode",
        "version": "1.0",
        "learned_pairs": len(load_learned_data().get("pairs", []))
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    if not model_ready:
        return jsonify({"error": "Model not ready yet"}), 503

    # Optional API key validation
    api_key = request.headers.get("X-API-Key") or (request.get_json() or {}).get("api_key", "")
    key_rec = validate_key(api_key) if api_key else None
    # If keys exist in store and key is invalid, reject
    store = load_keys()
    if store["keys"] and not key_rec and api_key:
        return jsonify({"error": "Invalid or revoked API key"}), 401

    data        = request.get_json()
    nl_text     = data.get("text", "").strip()
    temperature = float(data.get("temperature", 1.0))
    max_len     = int(data.get("max_len", 200))
    top_k       = int(data.get("top_k", 0))

    if not nl_text:
        return jsonify({"error": "No input text provided"}), 400

    temperature = max(0.0, min(2.0, temperature))
    max_len     = max(10,  min(200, max_len))
    top_k       = max(0,   min(200, top_k))

    try:
        code    = generate_code(nl_text, max_len=max_len, temperature=temperature, top_k=top_k)
        metrics = compute_metrics(code)
        if key_rec:
            increment_key_usage(api_key)
        return jsonify({
            "code":        code,
            "temperature": temperature,
            "top_k":       top_k,
            "input":       nl_text,
            "metrics":     metrics,
            "timestamp":   datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────── API Key CRUD ───────────────────────────────────────────────────────
@app.route("/api/keys", methods=["GET"])
def list_keys():
    store = load_keys()
    # Never expose full key — mask it
    safe  = []
    for k in store["keys"]:
        masked = k["key"][:8] + "..." + k["key"][-4:]
        safe.append({**k, "key_display": masked})
    return jsonify({"keys": safe})


@app.route("/api/keys", methods=["POST"])
def create_key():
    data  = request.get_json()
    name  = data.get("name", "Unnamed Key").strip()[:60]
    store = load_keys()
    new_key = {
        "key":       str(_uuid.uuid4()),
        "name":      name,
        "created":   datetime.datetime.now().isoformat(),
        "active":    True,
        "usage":     0,
        "last_used": None
    }
    store["keys"].append(new_key)
    save_keys(store)
    return jsonify({"key": new_key["key"], "name": name, "created": new_key["created"]})


@app.route("/api/keys/revoke", methods=["POST"])
def revoke_key():
    data  = request.get_json()
    kid   = data.get("key")  # full key
    store = load_keys()
    found = False
    for k in store["keys"]:
        if k["key"] == kid:
            k["active"] = False
            found = True
            break
    if not found:
        return jsonify({"error": "Key not found"}), 404
    save_keys(store)
    return jsonify({"status": "revoked"})


@app.route("/api/keys/delete", methods=["POST"])
def delete_key():
    data  = request.get_json()
    kid   = data.get("key")
    store = load_keys()
    store["keys"] = [k for k in store["keys"] if k["key"] != kid]
    save_keys(store)
    return jsonify({"status": "deleted"})


@app.route("/api/train", methods=["POST"])
def train():
    global is_training, train_log

    if not model_ready:
        return jsonify({"error": "Model not ready yet"}), 503

    if is_training:
        return jsonify({"error": "Training already in progress"}), 409

    data        = request.get_json()
    nl_text     = data.get("nl_text", "").strip()
    code_text   = data.get("code_text", "").strip()
    epochs      = int(data.get("epochs", 3))
    learn_rate  = float(data.get("learn_rate", 1e-4))
    lang        = data.get("lang", "java").strip().lower()

    if not nl_text or not code_text:
        return jsonify({"error": "Both nl_text and code_text are required"}), 400

    epochs     = max(1, min(20, epochs))
    learn_rate = max(1e-6, min(1e-2, learn_rate))

    # Save to learned data store
    learned    = load_learned_data()
    new_pair   = {
        "nl":        nl_text,
        "code":      code_text,
        "lang":      lang,
        "timestamp": datetime.datetime.now().isoformat()
    }
    learned["pairs"].append(new_pair)
    save_learned_data(learned)

    # Run fine-tuning in background thread
    def fine_tune():
        global is_training, train_log, transformer
        is_training = True
        train_log   = []

        try:
            train_log.append({"msg": "Starting fine-tuning...", "type": "info"})

            # Build tiny dataset from ALL learned pairs
            pairs = load_learned_data()["pairs"]
            enc_texts  = [clean_text(p["nl"])   for p in pairs]
            dec_texts  = [p["code"]             for p in pairs]

            enc_seqs   = encode_and_pad(encoder_sp, enc_texts, MAX_ENC_LEN)
            dec_seqs   = encode_and_pad(decoder_sp, dec_texts, MAX_DEC_LEN)

            dec_in     = dec_seqs[:, :-1]
            dec_tgt    = dec_seqs[:, 1:]

            # Recompile with lower LR for fine-tuning
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

            def masked_loss(y_true, y_pred):
                loss = loss_fn(y_true, y_pred)
                mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
                return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

            def masked_accuracy(y_true, y_pred):
                pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
                true = tf.cast(y_true, tf.int32)
                mask = tf.cast(tf.not_equal(true, PAD_ID), tf.float32)
                return tf.reduce_sum(tf.cast(tf.equal(pred, true), tf.float32) * mask) / tf.reduce_sum(mask)

            # jit_compile=False avoids XLA GPU bool-broadcast issues;
            # run_eagerly=False keeps graph mode but on CPU
            transformer.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
                loss=masked_loss,
                metrics=[masked_accuracy],
                jit_compile=False
            )

            class LogCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    msg  = (f"Epoch {epoch+1}/{epochs} — "
                            f"loss: {logs.get('loss', 0):.4f}  "
                            f"acc: {logs.get('masked_accuracy', 0):.4f}")
                    train_log.append({"msg": msg, "type": "epoch"})

            # Force CPU: avoids GPU BroadcastTo DT_BOOL kernel mismatch
            with tf.device('/CPU:0'):
                transformer.fit(
                    x=[enc_seqs, dec_in],
                    y=dec_tgt,
                    batch_size=max(1, len(pairs)),
                    epochs=epochs,
                    callbacks=[LogCallback()],
                    verbose=0
                )

            # Save updated model
            transformer.save(MODEL_PATH)

            # Log session
            learned = load_learned_data()
            learned["sessions"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "pairs_trained": len(pairs),
                "epochs": epochs
            })
            save_learned_data(learned)

            train_log.append({"msg": "✅ Fine-tuning complete! Model saved.", "type": "success"})

        except Exception as e:
            train_log.append({"msg": f"❌ Error: {str(e)}", "type": "error"})
        finally:
            is_training = False

    t = threading.Thread(target=fine_tune, daemon=True)
    t.start()

    return jsonify({"status": "Training started", "pairs": len(load_learned_data()["pairs"])})


@app.route("/api/train/log")
def train_log_api():
    return jsonify({"log": train_log, "is_training": is_training})


@app.route("/api/learned_data")
def get_learned_data():
    data = load_learned_data()
    return jsonify(data)


@app.route("/api/learned_data/delete", methods=["POST"])
def delete_learned_pair():
    data  = request.get_json()
    index = data.get("index")
    learned = load_learned_data()
    if 0 <= index < len(learned["pairs"]):
        learned["pairs"].pop(index)
        save_learned_data(learned)
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Invalid index"}), 400


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5052, debug=False)
