# 🤖 LLM Learning Journey

> Personal notes, code experiments, and projects while learning Large Language Models from scratch.

---

## 📚 Structure

```
LLM-Learning/
├── 01_basics/
│   ├── what_is_llm.md        # What are LLMs, how they work
│   └── tokenization.py       # Tokenization concepts + code
├── 02_transformers/
│   └── attention_notes.md    # Transformer architecture + attention
├── 03_langchain/
│   └── simple_chatbot.py     # Chatbot with memory using LangChain
└── README.md
```

---

## 🗺️ Learning Roadmap

| Module | Topic | Status |
|--------|-------|--------|
| 01 | What is an LLM | ✅ Done |
| 01 | Tokenization | ✅ Done |
| 02 | Transformer Architecture | ✅ Done |
| 02 | Attention Mechanism | ✅ Done |
| 03 | LangChain Basics | ✅ Done |
| 04 | RAG (Retrieval Augmented Generation) | 🔄 Next |
| 05 | Fine-tuning | 🔄 Upcoming |
| 06 | LLM Agents | 🔄 Upcoming |

---

## 🛠️ Setup

```bash
# Clone repo
git clone https://github.com/Abdul-Qadeerr/LLM-Learning.git
cd LLM-Learning

# Install dependencies
pip install transformers langchain langchain-openai python-dotenv torch

# Create .env file (optional — for OpenAI API)
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## 🚀 Quick Start

```bash
# Run tokenization examples
python 01_basics/tokenization.py

# Run chatbot (free HuggingFace model)
python 03_langchain/simple_chatbot.py
```

---

## 📖 Key Concepts Covered

**Basics**
- What LLMs are and how they work
- Tokenization — BPE, subword, word-level
- Context window and token limits

**Transformers**
- Encoder / Decoder / Encoder-Decoder architectures
- Self-attention and Multi-head attention
- Positional encoding
- Feed-forward layers, residual connections

**LangChain**
- Model initialization
- Prompt templates
- Chains — `prompt | model | parser`
- Conversation memory

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Attention is All You Need (Paper) | [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) |
| Andrej Karpathy — Let's build GPT | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| HuggingFace Course | [huggingface.co/learn](https://huggingface.co/learn) |
| LangChain Docs | [python.langchain.com](https://python.langchain.com) |
| 3Blue1Brown — Neural Networks | [YouTube Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |

---

## 👤 Author

**Abdul Qadeer**
- 10Pearls Data Science Intern
- Sukkur IBA University — BS Computer Science (2026)
- [GitHub](https://github.com/Abdul-Qadeerr) | [LinkedIn](https://linkedin.com/in/aqadeerr)

---


