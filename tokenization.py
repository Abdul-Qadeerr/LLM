# tokenization.py
# Understanding Tokenization in LLMs
# How text is converted to numbers before feeding to a model

# ── WHAT IS TOKENIZATION? ─────────────────────────────────────────────────────
# Tokenization = splitting text into smaller units (tokens)
# Tokens can be: words, subwords, characters, or bytes
# LLMs don't read text directly — they read token IDs (numbers)

# ── EXAMPLE 1: Basic Word Tokenization ────────────────────────────────────────
text = "I love machine learning"

# Simple word-level tokenization
word_tokens = text.split()
print("Word Tokens:", word_tokens)
# Output: ['I', 'love', 'machine', 'learning']

# Build a simple vocabulary
vocab = {word: idx for idx, word in enumerate(word_tokens)}
print("Vocabulary:", vocab)
# Output: {'I': 0, 'love': 1, 'machine': 2, 'learning': 3}

# Convert text to token IDs
token_ids = [vocab[word] for word in word_tokens]
print("Token IDs:", token_ids)
# Output: [0, 1, 2, 3]

print("\n" + "="*60 + "\n")

# ── EXAMPLE 2: Using HuggingFace Tokenizer ────────────────────────────────────
# pip install transformers
try:
    from transformers import AutoTokenizer

    # Load GPT-2 tokenizer (BPE based)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    text = "Large Language Models are amazing!"

    # Tokenize
    tokens = tokenizer.tokenize(text)
    print("GPT-2 Tokens:", tokens)

    # Convert to IDs
    token_ids = tokenizer.encode(text)
    print("Token IDs:", token_ids)

    # Decode back to text
    decoded = tokenizer.decode(token_ids)
    print("Decoded back:", decoded)

    # Token count
    print(f"Total tokens: {len(token_ids)}")

except ImportError:
    print("transformers not installed. Run: pip install transformers")

print("\n" + "="*60 + "\n")

# ── EXAMPLE 3: Subword Tokenization (BPE concept) ─────────────────────────────
# BPE = Byte Pair Encoding
# Used by GPT-2, GPT-3, GPT-4, LLaMA

# Why subword? 
# "unhappiness" → ["un", "happiness"] or ["un", "happy", "ness"]
# Handles rare/unknown words better than word-level tokenization

examples = [
    "unhappiness",      # common word split into subwords
    "ChatGPT",          # proper noun
    "tokenization",     # technical term
    "2024",             # number
    "Hello!",           # punctuation handling
]

try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("Subword Tokenization Examples:")
    print("-" * 40)
    for word in examples:
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.encode(word)
        print(f"'{word}' → {tokens} → {ids}")

except ImportError:
    print("Install transformers to see subword tokenization")

print("\n" + "="*60 + "\n")

# ── EXAMPLE 4: Context Window ─────────────────────────────────────────────────
# LLMs have a maximum token limit (context window)
# GPT-4: 128,000 tokens
# Claude 3: 200,000 tokens
# LLaMA 3: 8,192 tokens

# Rough estimate: 1 token ≈ 4 characters ≈ 0.75 words

def estimate_tokens(text):
    """Rough token count estimate"""
    return len(text) // 4

sample_text = "The quick brown fox jumps over the lazy dog."
estimated = estimate_tokens(sample_text)
print(f"Text: '{sample_text}'")
print(f"Characters: {len(sample_text)}")
print(f"Estimated tokens: ~{estimated}")

print("\n" + "="*60 + "\n")

# ── KEY TAKEAWAYS ─────────────────────────────────────────────────────────────
print("""
KEY TAKEAWAYS:
--------------
1. LLMs work with tokens, not raw text
2. Tokens can be subwords — "playing" → ["play", "ing"]  
3. Each token gets a unique ID number
4. Context window = max tokens model can process at once
5. More tokens = more cost in API calls
6. GPT-4 uses tiktoken library for tokenization
""")
