# 🧠 Emotional vs Logical Chatbot

A conversational AI system built with **LangGraph**, **LangChain**, and **Gemini 2.0**, designed to intelligently classify user input as either emotional or logical and route the message to a specialized agent (therapist or logical assistant) accordingly.

---

## ✨ Features

- ✅ Classifies input messages as **emotional** or **logical**
- 🧘 Provides empathetic responses for emotionally driven inputs
- 🧠 Offers factual, logical answers for informational queries
- 🧠 Memory-enabled: Remembers context like name, emotions, and past answers across turns using InMemorySaver
- 🔁 Conversational state maintained with LangGraph and add_messages
- 🧰 Works with both open-source and closed-source models (e.g., Gemini 2.0, GPT-4, Mistral)
- 🔁 Stateful conversation loop using LangGraph
- 🔐 Environment variable support with `python-dotenv`

---

<img width="686" alt="image" src="https://github.com/user-attachments/assets/f8706467-bcce-4798-8e91-29a16c5fe4e0" />


## 🗂 Project Structure (UV Compatible)


## 🧠 Sample Conversation (with Memory)
```
You: hi  
Assistant: Hello.  
You: I'm Vijay  
Assistant: Acknowledged. You are Vijay.  
You: what is my name  
Assistant: Your name is Vijay. You stated, "I'm Vijay."  
You: I'm feeling down  
Assistant: Oh, Vijay, I'm truly sorry to hear you're feeling down. Can you tell me a little more about what's making you feel this way?  
You: because I lost my job  
Assistant: Losing your job... Vijay, that's a really tough experience. It's completely understandable that you're feeling down...  
You: thanks buddy  
Assistant: You're very welcome, Vijay. I'm glad I could offer some support.  
You: I want to get job any plan  
Assistant: Let's formulate a plan... (gives step-by-step job search strategy)
```
✅ This dynamic behavior is powered by LangGraph memory + message classification


## 🗂 Project Structure
```
📁 langgraph_agent_logical_emotional
├── main.py              # Core chatbot with LangGraph and message routing
├── .env                 # API keys & environment config
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
```




