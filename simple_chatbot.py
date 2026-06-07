# simple_chatbot.py
# Building a Simple Chatbot using LangChain + OpenAI
# LangChain = framework that makes building LLM apps easier

# ── INSTALLATION ──────────────────────────────────────────────────────────────
# pip install langchain langchain-openai python-dotenv

# ── WHAT IS LANGCHAIN? ────────────────────────────────────────────────────────
# LangChain is a framework for building LLM-powered applications.
# It provides:
# - Easy model connection (OpenAI, Anthropic, HuggingFace)
# - Prompt templates
# - Memory (conversation history)
# - Chains (connect multiple steps)
# - Agents (LLM + tools)
# - RAG (connect LLM to your documents)

import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# ── EXAMPLE 1: Simple LLM Call ────────────────────────────────────────────────
def example_1_simple_call():
    """Most basic LLM call using LangChain"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        # Initialize model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,          # 0 = deterministic, 1 = creative
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Send a message
        response = llm.invoke([HumanMessage(content="What is AI in one sentence?")])
        print("Response:", response.content)

    except ImportError:
        print("Run: pip install langchain langchain-openai")
    except Exception as e:
        print(f"Error: {e}")


# ── EXAMPLE 2: Prompt Template ────────────────────────────────────────────────
def example_2_prompt_template():
    """Using PromptTemplate to structure inputs"""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

        # Define a reusable prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that explains {topic} simply."),
            ("human", "{question}")
        ])

        # Create a chain: prompt → llm
        chain = prompt | llm

        # Invoke with variables
        response = chain.invoke({
            "topic": "machine learning",
            "question": "What is overfitting?"
        })

        print("Response:", response.content)

    except ImportError:
        print("Run: pip install langchain langchain-openai")
    except Exception as e:
        print(f"Error: {e}")


# ── EXAMPLE 3: Chatbot with Memory ────────────────────────────────────────────
def example_3_chatbot_with_memory():
    """
    A chatbot that remembers conversation history.
    Without memory: each message is independent.
    With memory: model remembers previous messages.
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage

        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

        # Prompt with message history placeholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        chain = prompt | llm

        # Conversation history (manual memory)
        history = []

        print("Simple Chatbot (type 'quit' to exit)")
        print("-" * 40)

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Get response
            response = chain.invoke({
                "history": history,
                "input": user_input
            })

            ai_reply = response.content
            print(f"AI: {ai_reply}\n")

            # Add to history
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=ai_reply))

            # Keep last 10 messages (5 exchanges) to manage token usage
            if len(history) > 10:
                history = history[-10:]

    except ImportError:
        print("Run: pip install langchain langchain-openai")
    except Exception as e:
        print(f"Error: {e}")


# ── EXAMPLE 4: Free Alternative (HuggingFace) ────────────────────────────────
def example_4_free_huggingface():
    """
    Use HuggingFace models for FREE (no API key needed for some models)
    Good for learning without paying for OpenAI
    """
    try:
        from transformers import pipeline

        # Free, runs locally
        generator = pipeline(
            "text-generation",
            model="distilgpt2",      # Small, fast model — good for learning
            max_new_tokens=50,
            pad_token_id=50256
        )

        prompt = "Machine learning is"
        result = generator(prompt)
        print("Generated:", result[0]['generated_text'])

    except ImportError:
        print("Run: pip install transformers torch")
    except Exception as e:
        print(f"Error: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Simple Chatbot Examples")
    print("=" * 60)

    print("\n[Example 4 - Free HuggingFace Model]")
    example_4_free_huggingface()

    # Uncomment below if you have OpenAI API key in .env
    # print("\n[Example 1 - Simple LLM Call]")
    # example_1_simple_call()

    # print("\n[Example 2 - Prompt Template]")
    # example_2_prompt_template()

    # print("\n[Example 3 - Chatbot with Memory]")
    # example_3_chatbot_with_memory()


# ── LANGCHAIN KEY CONCEPTS SUMMARY ───────────────────────────────────────────
"""
LangChain Building Blocks:
--------------------------
1. Model         → ChatOpenAI, HuggingFace, Anthropic
2. Prompt        → ChatPromptTemplate, PromptTemplate
3. Chain         → prompt | model | output_parser
4. Memory        → Store conversation history
5. Output Parser → Convert model output to specific format
6. Tools         → Search, Calculator, Code execution
7. Agents        → LLM decides which tools to use
8. RAG           → Retrieval Augmented Generation

Learning Path:
--------------
Week 1: Simple calls + Prompt templates
Week 2: Memory + Chains
Week 3: Tools + Agents
Week 4: RAG (connect LLM to your documents)
"""
