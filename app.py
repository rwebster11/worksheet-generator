import os
import anthropic
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import traceback # Import traceback for detailed error logging
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re # For parsing YouTube URL
# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Initialize the Anthropic Client (More robust initialization)
try:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not found in environment variables.")
        client = None
    else:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialized successfully.")
        # Optional: Add a test call here if you want to verify during startup
        # try:
        #     client.messages.create(model="claude-3-haiku-20240307", max_tokens=10, messages=[{"role": "user", "content": "Hello"}])
        #     print("Test API call successful.")
        # except Exception as api_err:
        #     print(f"Warning: Test API call failed during initialization: {api_err}")

except Exception as e:
    print(f"Error initializing Anthropic client: {e}")
    client = None # Ensure client is None if initialization fails

# Use Opus model as decided previously
ANTHROPIC_MODEL_NAME = "claude-3-opus-20240229"

# --- Prompt Function ---
def create_gap_fill_prompt(topic, grade_level="middle school"):
    """Creates a more forceful prompt emphasizing grade level adaptation."""
    prompt = f"""You are an expert teacher creating educational resources tailored for a specific audience.
Your task is to generate a gap-fill (cloze) activity about the topic '{topic}'.

**Target Audience: {grade_level} students.**

**CRITICAL INSTRUCTIONS:**
1.  **Adapt Content to Audience:** You **MUST** adjust the vocabulary complexity, sentence structure, and depth of concepts presented to be appropriate for **{grade_level} students**. This is a primary requirement.
    *   For 'elementary school', use very simple terms, short sentences, and focus on the most basic ideas. Explain concepts clearly.
    *   For 'middle school', use standard, clear language and introduce core concepts accurately.
    *   For 'high school', use more specific terminology, assume some background knowledge, and employ slightly more complex sentences.
    *   For 'university' or 'general adult', use precise, potentially academic or professional language, assume significant prior knowledge, and handle nuanced concepts.
2.  **Sentence Content:** Create 5-10 unique sentences. Each sentence **MUST** focus on a **different** important aspect or key term of the topic '{topic}'. Avoid rephrasing the same core idea.
3.  **Blank Creation:** In each sentence, identify the single most important key term or concept specific to that sentence's point and replace it with '_________'.
4.  **Blank Variety:** **Each blank MUST be fillable with a DIFFERENT word.** Do NOT reuse the same answer word for multiple blanks. This is crucial for testing diverse vocabulary. Using the same word in two or more blanks is forbidden.
5.  **Answer Key:** After the sentences, provide a numbered list titled 'Answer Key:' listing the single word removed for each blank in the correct order.
6.  **Output Format:** Output ONLY the worksheet sentences and the Answer Key. No introductory phrases, explanations, conversational text, or titles other than 'Answer Key:'.

Topic: {topic}

Worksheet:
"""
    return prompt
    # Removed the extra 'return prompt' that was here

# --- Route to Serve the Frontend HTML ---
@app.route('/')
def serve_index():
    """Serves the index.html file."""
    print("Serving index.html")
    return send_from_directory('.', 'index.html')

# --- Route to Handle Worksheet Generation ---
@app.route('/generate_worksheet', methods=['POST'])
def generate_worksheet_route():
    """Handles POST requests to generate a worksheet."""
    print("Received request at /generate_worksheet")

    if client is None:
        print("Error: Anthropic client not initialized.")
        return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500

    if not request.is_json:
        print("Error: Request is not JSON.")
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        # Consistently indent all lines inside the 'try' block
        data = request.get_json()
        # Use consistent 4-space indentation for this block
        topic = data.get('topic')
        grade_level = data.get('grade_level', 'middle school') # Default if missing

        # Log received data (Corrected indentation)
        print(f"Received data: {data}")

        if not topic:
            # Indent block under 'if' consistently
            print("Error: 'topic' missing in request data.")
            return jsonify({'status': 'error', 'message': 'Missing "topic" in request data'}), 400

        # Log the topic and grade level correctly (Corrected indentation and removed extra parenthesis)
        print(f"Received topic: '{topic}', Grade Level: '{grade_level}'")

        # Create the prompt (Corrected indentation)
        prompt_content = create_gap_fill_prompt(topic, grade_level)

        # --- Call Anthropic API --- (Corrected indentation)
        print(f"Sending request to Anthropic API (Model: {ANTHROPIC_MODEL_NAME})...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=700, # Increased max_tokens slightly for potentially longer Opus responses
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        )
        print("Received response from Anthropic API.")

        # Extract the generated text (Corrected indentation)
        generated_text = ""
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            # Correctly indent line inside 'if'
            generated_text = message.content[0].text
        else:
            # Correctly indent lines inside 'else'
            print(f"Warning: Unexpected API response structure or empty content. Response: {message}")
            return jsonify({'status': 'error', 'message': 'Failed to parse content from API response.'}), 500

        # Log content length (Corrected indentation)
        print(f"Generated content length: {len(generated_text)} characters")

        # --- Return the Result to the Frontend --- (Corrected indentation)
        return jsonify({
            'status': 'success',
            'worksheet_content': generated_text.strip()
        })

    # --- Specific Error Handling for Anthropic API ---
    except anthropic.APIConnectionError as e:
        # Correctly indent lines inside 'except'
        print(f"API Connection Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e:
        # Correctly indent lines inside 'except'
        print(f"API Rate Limit Error: {e}")
        return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    except anthropic.APIStatusError as e:
        # Correctly indent lines inside 'except'
        print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
        error_message = f'AI service error (Status {e.status_code})'
        try:
            # Indent lines inside nested 'try'
            error_details = e.response.json()
            error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception:
            # Indent line inside nested 'except'
            error_message += f": {e.response.text}"
        # Ensure return is indented relative to the outer 'except'
        return jsonify({'status': 'error', 'message': error_message}), e.status_code
    # --- General Error Handling ---
    except Exception as e:
        # Correctly indent lines inside 'except'
        print(f"An unexpected error occurred in /generate_worksheet: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f'An internal server error occurred.'}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Correctly indent line inside 'if'
    app.run(host='127.0.0.1', port=5001, debug=True)