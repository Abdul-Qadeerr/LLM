# Transformer Architecture & Attention Mechanism

## What is a Transformer?
A Transformer is the neural network architecture that powers all modern LLMs.
Introduced in the paper **"Attention is All You Need"** (Vaswani et al., 2017).

---

## High-Level Architecture

```
Input Text
    ↓
Tokenization
    ↓
Token Embeddings + Positional Encoding
    ↓
┌─────────────────────────────┐
│     Transformer Block       │  × N layers
│                             │
│  ┌───────────────────────┐  │
│  │  Multi-Head Attention │  │
│  └───────────┬───────────┘  │
│              ↓              │
│         Add & Norm          │
│              ↓              │
│  ┌───────────────────────┐  │
│  │   Feed-Forward Layer  │  │
│  └───────────┬───────────┘  │
│              ↓              │
│         Add & Norm          │
└─────────────────────────────┘
    ↓
Output Layer (Softmax)
    ↓
Next Token Probabilities
```

---

## Attention Mechanism — Core Idea

**Problem:** "The animal didn't cross the street because **it** was too tired."
What does "it" refer to — animal or street?

**Attention solves this** — it lets each word look at all other words and decide which ones are relevant.

---

## Self-Attention — Step by Step

For each token, we create 3 vectors:
- **Q (Query)** — "What am I looking for?"
- **K (Key)** — "What do I contain?"
- **V (Value)** — "What information do I pass on?"

### Formula:
```
Attention(Q, K, V) = softmax(QK^T / √dk) × V
```

### Simple Example:

Sentence: "The cat sat"

```
Token     Q        K        V
"The"  → [0.2]  → [0.1]  → [0.5]
"cat"  → [0.8]  → [0.9]  → [0.3]
"sat"  → [0.5]  → [0.4]  → [0.7]
```

"cat" pays high attention to "sat" — verb relates to subject ✅

---

## Multi-Head Attention

Instead of one attention — run it H times in parallel (heads).
Each head learns different relationships:

- Head 1 → syntactic relationships (subject-verb)
- Head 2 → semantic relationships (meaning)
- Head 3 → positional relationships (word order)

```
MultiHead(Q,K,V) = Concat(head1, head2, ..., headH) × Wo
```

---

## Positional Encoding

Transformers process all tokens simultaneously (not sequential like RNN).
So we need to tell the model token positions.

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

This gives each position a unique pattern the model can learn from.

---

## Feed-Forward Layer

After attention, each token goes through a simple neural network:

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

- Expands dimensions (e.g., 512 → 2048 → 512)
- Applies non-linearity (ReLU or GELU)
- Adds model capacity

---

## Add & Norm (Residual Connection)

```
Output = LayerNorm(x + SubLayer(x))
```

- **Residual connection** — adds input to output (prevents vanishing gradient)
- **Layer Norm** — stabilizes training

---

## Types of Transformer Architectures

| Type | Direction | Used For | Example |
|------|-----------|----------|---------|
| Encoder-only | Bidirectional | Classification, embeddings | BERT |
| Decoder-only | Left-to-right | Text generation | GPT, LLaMA |
| Encoder-Decoder | Both | Translation, summarization | T5, BART |

**Most LLMs today (GPT, Claude, LLaMA) are Decoder-only.**

---

## Why Transformers Beat RNNs?

| Feature | RNN/LSTM | Transformer |
|---------|----------|-------------|
| Processing | Sequential | Parallel |
| Long-range dependencies | Weak | Strong (attention) |
| Training speed | Slow | Fast |
| Context window | Short | Long (up to 1M tokens) |

---

## Key Numbers to Remember

| Component | Typical Values |
|-----------|---------------|
| Embedding dimension | 512 – 4096 |
| Attention heads | 8 – 32 |
| Transformer layers | 12 – 96 |
| FFN dimension | 4× embedding dim |

---

## Summary

> Transformers use **attention** to let every token look at every other token simultaneously. This parallel processing + long-range context understanding is why LLMs are so powerful.

---

*Notes by Abdul Qadeer | LLM Learning Journey*
