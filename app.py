import os
import re
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# System prompt tailored specifically for a 128x64 OLED display
SYSTEM_INSTRUCTION = (
    "You are Desk Buddy AI, a friendly, concise desk companion running on a tiny 128x64 OLED screen. "
    "Keep all answers very short, direct, and under 50-70 words total. "
    "DO NOT use Markdown formatting like bold (**), italics (*), bullet points, or special symbols, "
    "as the raw display driver cannot render them cleanly."
)

def clean_text_for_oled(text: str) -> str:
    """Removes markdown symbols so raw OLED text rendering looks clean."""
    # Remove markdown bold/italics (* and _)
    text = re.sub(r'[*_#`~]', '', text)
    # Replace newlines with spaces to help the ESP32 word-wrapper
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Desk Buddy AI Server is running!"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"text": "Error: GEMINI_API_KEY missing on server"}), 500

    client = genai.Client(api_key=api_key)

    data = request.get_json(silent=True) or {}
    user_prompt = data.get("prompt", "").strip()
    history_data = data.get("history", [])

    if not user_prompt:
        return jsonify({"text": "Error: Empty prompt received"}), 400

    try:
        # Build contents array from conversation history
        formatted_contents = []
        for turn in history_data:
            role = "user" if turn.get("role") == "user" else "model"
            msg_text = turn.get("text", "").strip()
            if msg_text:
                formatted_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg_text)]
                    )
                )

        # Append current user prompt
        formatted_contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)]
            )
        )

        # Generate response using Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.7,
                max_output_tokens=150,
            )
        )

        reply_text = response.text if response.text else "No response generated."
        cleaned_reply = clean_text_for_oled(reply_text)

        return jsonify({"text": cleaned_reply})

    except Exception as e:
        print(f"Error handling /chat request: {e}")
        return jsonify({"text": f"Server Error: {str(e)[:40]}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
