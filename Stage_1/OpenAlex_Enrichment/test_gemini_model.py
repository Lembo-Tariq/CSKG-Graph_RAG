import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)


"""
    Gemini models:
        models/gemini-2.5-flash
        models/gemini-2.5-pro
        models/gemini-2.0-flash
        models/gemini-2.0-flash-001
        models/gemini-2.0-flash-lite-001
        models/gemini-2.0-flash-lite
    
    We will continue with Groq because:
        Groq llama-3.1-8b-instant is better for this project because:

        Already integrated and working in project
        Extremely fast (fastest inference available)
        Very generous free tier (14,400 requests/day)
        No package conflicts — already seen Google's packages cause version headaches
        Consistent — we use Groq everywhere else in the project already

        Gemini 2.0 Flash Lite is better if:

        You need stronger reasoning or longer context
        You want to use Google's ecosystem consistently
        The task requires more nuanced language generation

    Claude's Answer:
        My recommendation
        Stick with Groq for this project for two reasons:

        Consistency — your RAG already uses Groq, your app uses Groq, keeping everything in one ecosystem makes the project cleaner
        No dependency headaches — every time we use Google packages we run into version conflicts
"""