import os
import anthropic
# Near other imports
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone # Added timezone for UTC awareness
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import traceback # Import traceback for detailed error logging
import logging # Import standard logging

# Configure basic logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

# Load environment variables from .env file
logging.info("Attempting to load .env file...")
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Explicit path
found_dotenv = load_dotenv(dotenv_path=dotenv_path, verbose=True) # Be verbose
logging.info(f".env file found and loaded: {found_dotenv}")
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re # For parsing YouTube URL
# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application

app = Flask(__name__)

# --- Database Configuration ---
# Get the absolute path of the directory where app.py is located
basedir = os.path.abspath(os.path.dirname(__file__))
# Define the path for the SQLite database file
db_path = os.path.join(basedir, 'worksheet_library.db')
# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
# Disable modification tracking, it's not needed and uses extra memory
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# --- End Database Configuration ---

# Initialize the SQLAlchemy extension
db = SQLAlchemy(app)
# --- Database Model Definition ---
class GeneratedItem(db.Model):
    __tablename__ = 'generated_items' # Explicit table name (optional, but good practice)

    id = db.Column(db.Integer, primary_key=True) # Auto-incrementing primary key
    item_type = db.Column(db.String(50), nullable=False) # 'gap_fill', 'youtube_comprehension', etc.
    source_topic = db.Column(db.String(250), nullable=True) # Topic used for generation (nullable for non-topic items)
    source_url = db.Column(db.String(500), nullable=True) # URL used for generation (nullable for topic items)
    grade_level = db.Column(db.String(50), nullable=False) # e.g., 'middle school'
    content_html = db.Column(db.Text, nullable=False) # The generated/edited HTML content
    # Use timezone.utc for database consistency
    creation_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_modified_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    # Potential future fields:
    # tags = db.Column(db.String(200), nullable=True) # Comma-separated tags? Or a separate table later.
    # description = db.Column(db.String(500), nullable=True) # User-added description

    def __repr__(self):
        # Helpful representation for debugging
        return f'<GeneratedItem id={self.id} type={self.item_type} topic="{self.source_topic}" url="{self.source_url}">'

# --- End Database Model Definition ---

# Initialize the Anthropic Client (More robust initialization)
client = None
try:
    logging.info("Attempting to initialize Anthropic client...")
    api_key_from_env = os.getenv("ANTHROPIC_API_KEY")

    if not api_key_from_env:
        logging.error("CRITICAL: ANTHROPIC_API_KEY not found in environment variables.")
    else:
        logging.info("ANTHROPIC_API_KEY found. Initializing client...")

        # *** ADD PROXY CONFIGURATION FOR PYTHONANYWHERE FREE TIER ***
        proxy_url = "http://proxy.server:3128"
        proxies = {
            "http": proxy_url,
            "https": proxy_url,
        }

        client = anthropic.Anthropic(
            api_key=api_key_from_env,
            proxies=proxies # <-- Pass the proxy configuration here
        )
        logging.info("Anthropic client object CREATED successfully (with proxy).")
        # ... (optional test call) ...
except Exception as e:
    # Log any exception during the client initialization process
    logging.error(f"CRITICAL: Exception during Anthropic client initialization: {e}", exc_info=True) # Log traceback too
    client = None # Ensure client is None on error

# Check client status immediately after initialization block
if client is None:
    logging.warning("Anthropic client is None immediately after initialization block.")
else:
    logging.info("Anthropic client appears to be initialized successfully after try/except block.")


# Use Opus model as decided previously
ANTHROPIC_MODEL_NAME = "claude-3-opus-20240229"

# --- Prompt Function ---
def create_mcq_prompt(topic, grade_level, num_questions=5):
    """Creates the prompt for Claude to generate multiple-choice questions."""
    # This prompt needs to be very specific about the output format
    prompt = f"""You are an expert curriculum developer creating assessment questions.
Generate {num_questions} multiple-choice questions (MCQs) about the topic '{topic}'.
The target audience is {grade_level} students.

**CRITICAL INSTRUCTIONS for Output Format:**
1.  **Question Generation:** Create {num_questions} distinct MCQs covering different important aspects of the topic. Adapt vocabulary and complexity for the {grade_level}.
2.  **Options:** For each question, provide exactly four options: one correct answer (labeled A, B, C, or D) and three plausible distractors.
3.  **Formatting:**
    *   Present each question clearly, starting with a number (e.g., "1.", "2.").
    *   List the options below each question, labeled "A.", "B.", "C.", "D.".
    *   **DO NOT** indicate the correct answer within the question/option list itself.
4.  **Answer Key:** After ALL questions and their options are listed, provide a separate section titled exactly "Answer Key:".
5.  **Key Format:** In the Answer Key section, list the question number and the correct option letter (e.g., "1. C", "2. A", "3. B", etc.), each on a new line.
6.  **Strict Output:** Output ONLY the numbered questions with their A/B/C/D options, followed by the "Answer Key:" section and the key itself. No extra text, introductions, or explanations.

Topic: {topic}
Audience: {grade_level}

MCQs:
"""
    return prompt
def create_text_block_prompt(topic, grade_level, focus=None):
    """Creates the prompt for Claude to generate a block of text."""
    # Instructions can be added based on 'focus' or other parameters later
    focus_instruction = f"Focus on: {focus}" if focus else "Provide a general overview."

    prompt = f"""You are a helpful assistant generating informational content.
Write a clear and concise text block about the topic '{topic}'.
The target audience is {grade_level} students.
{focus_instruction}
Aim for 1-3 paragraphs unless specified otherwise.
Ensure the language is appropriate for the specified grade level.

**Output ONLY the text block itself.** No titles, introductions, or extra formatting other than paragraphs.

Topic: {topic}
Audience: {grade_level}

Text Block:
"""
    return prompt
def create_true_false_prompt(topic, grade_level, num_statements=8):
    """Creates the prompt for Claude to generate True/False statements."""
    prompt = f"""You are an expert educator creating assessment materials.
Generate {num_statements} True/False statements about the core concepts of the topic '{topic}'.
The target audience is {grade_level} students.

**CRITICAL INSTRUCTIONS:**
1.  **Statement Generation:** Create {num_statements} clear statements. Ensure a mix of reasonably challenging statements that are definitively TRUE and definitively FALSE based on common knowledge for the {grade_level} audience regarding the topic '{topic}'. Avoid ambiguity or opinion-based statements.
2.  **Formatting:** Present each statement clearly, starting with a number (e.g., "1.", "2.").
3.  **Answer Key:** After ALL statements are listed, provide a separate section titled exactly "Answer Key:".
4.  **Key Format:** In the Answer Key section, list the statement number followed by the word "True" or "False" (e.g., "1. True", "2. False", "3. True", etc.), each on a new line.
5.  **Strict Output:** Output ONLY the numbered statements, followed by the "Answer Key:" section and the key itself. No extra text, introductions, or explanations.

Topic: {topic}
Audience: {grade_level}

True/False Statements:
"""
    return prompt
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

def extract_video_id(url):
    """Extracts the YouTube video ID from various URL formats."""
    # Regex patterns to match standard YouTube URLs and short URLs
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',  # Standard URL
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',          # Short URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})', # Embed URL
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',     # V URL
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1) # Return the first capture group (the ID)
    return None # Return None if no match found

def create_comprehension_prompt(transcript_text, num_questions=5):
    """Creates the prompt for Claude to generate comprehension questions from a transcript."""
    # Use the num_questions variable in the prompt f-string
    prompt = f"""You are an expert educator. Based on the following video transcript, please generate {num_questions} insightful comprehension questions that test understanding of the key information presented.

**Instructions for Questions:**
1.  Ensure questions cover different aspects of the transcript (e.g., main ideas, specific details, inferences if possible).
2.  Phrase questions clearly and concisely.
3.  The questions should be suitable for someone who has just watched the video (assume context from the transcript).
4.  Do NOT provide the answers to the questions.
5.  Format the output as a numbered list.
6.  Output ONLY the numbered list of questions, nothing else (no preamble like "Here are the questions:").

**Video Transcript:**
--- START TRANSCRIPT ---
{transcript_text}
--- END TRANSCRIPT ---

**{num_questions} Comprehension Questions:**
"""
    return prompt
# ---  create_short_answer_prompt function ---
def create_short_answer_prompt(topic, grade_level, num_questions=5):
    """Creates the prompt for Claude to generate short answer questions AND answer key points."""
    prompt = f"""You are an expert educator designing formative assessments.
Generate {num_questions} open-ended short answer questions about the key aspects of the topic '{topic}'.
Also provide a brief answer key outlining the expected points or a model answer for each question.
The target audience is {grade_level} students.

**CRITICAL INSTRUCTIONS:**
1.  **Question Generation:** Create {num_questions} distinct questions that require students to recall, explain, or briefly analyze information related to '{topic}'. Questions should encourage answers of 1-3 sentences. Adapt vocabulary and complexity for the {grade_level}.
2.  **Question Formatting:** Present each question clearly, starting with a number (e.g., "1.", "2."). List each question on a new line.
3.  **Answer Key Generation:** After ALL questions are listed, provide a separate section titled exactly "Answer Key:".
4.  **Key Content:** For each question number in the Answer Key section, provide either:
    *   A bulleted list of the key facts/points expected in a good answer.
    *   OR a brief model answer (1-2 sentences).
    Keep the answer key concise and focused on grading criteria.
5.  **Key Formatting:** Start each answer key item with the corresponding question number (e.g., "1.", "2.").
6.  **Strict Output:** Output ONLY the numbered list of questions, followed by the "Answer Key:" section and the key itself. No extra text, introductions, or explanations.

Topic: {topic}
Audience: {grade_level}

Short Answer Questions:
[Generate Questions Here Following Format]

Answer Key:
[Generate Answer Key Here Following Format]
"""
    return prompt
# --- Route to List Saved Items ---
@app.route('/list_items', methods=['GET'])
def list_items_route():
    """Handles GET requests to retrieve saved items from the database."""
    print("Received request at /list_items")
    try:
        # Query the database - get most recent 20 items for now
        # Order by last modified date descending
        items = GeneratedItem.query.order_by(GeneratedItem.last_modified_date.desc()).limit(20).all()

        # Convert items to a list of dictionaries for JSON serialization
        items_list = []
        for item in items:
            items_list.append({
                'id': item.id,
                'item_type': item.item_type,
                'source_topic': item.source_topic,
                'source_url': item.source_url,
                'grade_level': item.grade_level,
                # Optionally include creation/modified date (convert to string)
                'last_modified': item.last_modified_date.isoformat(),
                # Avoid sending full content_html in the list view for brevity
                # 'content_preview': item.content_html[:100] + '...' # Example preview
            })

        return jsonify({'status': 'success', 'items': items_list})

    except Exception as e:
        print(f"Error retrieving items from database: {e}")
        print(traceback.format_exc())
        # Avoid sending detailed DB errors to frontend in production
        return jsonify({'status': 'error', 'message': 'Error retrieving items from library.'}), 500

# --- End of List Items Route ---
# --- Route for True/False Statement Generation ---
@app.route('/generate_true_false', methods=['POST'])
def generate_true_false_route():
    """Handles POST requests to generate True/False statements."""
    print("Received request at /generate_true_false")

    if client is None: # Check Anthropic client
        return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: # Check request format
         return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        topic = data.get('topic')
        grade_level = data.get('grade_level', 'middle school')
        num_statements_req = data.get('num_statements', 8) # Default to 8 statements

        # --- Validation ---
        if not topic:
            return jsonify({'status': 'error', 'message': 'Missing "topic" in request data'}), 400
        try:
            num_statements = int(num_statements_req)
            if not 3 <= num_statements <= 20: # Sensible range for T/F
                raise ValueError("Number of statements must be between 3 and 20.")
        except (ValueError, TypeError):
             return jsonify({'status': 'error', 'message': 'Invalid number of statements specified (must be an integer between 3 and 20).'}), 400

        print(f"Received T/F request: Topic='{topic}', Grade='{grade_level}', NumStatements={num_statements}")

        # --- Create Prompt ---
        prompt_content = create_true_false_prompt(topic, grade_level, num_statements)

        # --- Call Anthropic API ---
        print(f"Sending T/F request to Anthropic API (Model: {ANTHROPIC_MODEL_NAME})...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=1000 + (num_statements * 50), # Estimate tokens
            temperature=0.6, # Maybe slightly less creative for T/F?
            messages=[{ "role": "user", "content": prompt_content }]
        )
        print("Received response from Anthropic API.")

        # --- Extract Result ---
        generated_content = ""
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            generated_content = message.content[0].text
        else:
             print(f"Warning: Unexpected API response structure or empty content. Response: {message}")
             return jsonify({'status': 'error', 'message': 'Failed to parse content from AI response.'}), 500

        print(f"Generated T/F content length: {len(generated_content)} chars")

        # --- Return Result (use a distinct key) ---
        return jsonify({
            'status': 'success',
            'true_false_content': generated_content.strip() # Use 'true_false_content' key
        })

    # --- Error Handling (reuse existing handlers) ---
    except ValueError as ve: # Catch our validation errors
         print(f"Validation Error: {ve}")
         return jsonify({'status': 'error', 'message': str(ve)}), 400
    except anthropic.APIConnectionError as e: # ... Copy other handlers ...
        print(f"API Connection Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e:
        print(f"API Rate Limit Error: {e}")
        return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    except anthropic.APIStatusError as e:
        print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
        error_message = f'AI service error (Status {e.status_code})'
        try: error_details = e.response.json(); error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception: error_message += f": {e.response.text}"
        return jsonify({'status': 'error', 'message': error_message}), e.status_code
    except Exception as e:
        print(f"An unexpected error occurred in /generate_true_false: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500

# --- End of True/False Route ---
# --- Route for Multiple Choice Question Generation ---
@app.route('/generate_mcq', methods=['POST'])
def generate_mcq_route():
    """Handles POST requests to generate Multiple Choice Questions."""
    print("Received request at /generate_mcq")

    if client is None: # Check Anthropic client
        # ... (keep existing error handling) ...
        return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: # Check request format
        # ... (keep existing error handling) ...
         return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        topic = data.get('topic')
        grade_level = data.get('grade_level', 'middle school') # Default grade level
        num_questions_req = data.get('num_questions', 5) # Default number

        # --- Validation ---
        if not topic:
            return jsonify({'status': 'error', 'message': 'Missing "topic" in request data'}), 400
        try:
            num_questions = int(num_questions_req)
            if not 2 <= num_questions <= 15: # Adjust range as needed
                raise ValueError("Number of questions must be between 2 and 15.")
        except (ValueError, TypeError):
             return jsonify({'status': 'error', 'message': 'Invalid number of questions specified (must be an integer between 2 and 15).'}), 400

        print(f"Received MCQ request: Topic='{topic}', Grade='{grade_level}', NumQuestions={num_questions}")

        # --- Create Prompt ---
        prompt_content = create_mcq_prompt(topic, grade_level, num_questions)

        # --- Call Anthropic API ---
        print(f"Sending MCQ request to Anthropic API (Model: {ANTHROPIC_MODEL_NAME})...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=1500 + (num_questions * 100), # Generous token estimate for questions + options
            temperature=0.7, # Adjust as needed
            messages=[{ "role": "user", "content": prompt_content }]
        )
        print("Received response from Anthropic API.")

        # --- Extract Result ---
        generated_content = ""
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            generated_content = message.content[0].text
        else:
             # ... (keep existing error handling for bad response structure) ...
             print(f"Warning: Unexpected API response structure or empty content. Response: {message}")
             return jsonify({'status': 'error', 'message': 'Failed to parse content from AI response.'}), 500

        print(f"Generated MCQ content length: {len(generated_content)} chars")

        # --- Return Result (use a distinct key) ---
        return jsonify({
            'status': 'success',
            'mcq_content': generated_content.strip() # Use 'mcq_content' key
        })

    # --- Error Handling (reuse existing handlers) ---
    except ValueError as ve: # Catch our validation errors
         print(f"Validation Error: {ve}")
         return jsonify({'status': 'error', 'message': str(ve)}), 400
    # ... include the existing except blocks for anthropic errors and general Exception ...
    except anthropic.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e:
        print(f"API Rate Limit Error: {e}")
        return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    except anthropic.APIStatusError as e:
        print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
        # ... (error message extraction logic from previous routes) ...
        error_message = f'AI service error (Status {e.status_code})'
        try:
            error_details = e.response.json()
            error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception:
            error_message += f": {e.response.text}"
        return jsonify({'status': 'error', 'message': error_message}), e.status_code
    except Exception as e:
        print(f"An unexpected error occurred in /generate_mcq: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500

# --- End of MCQ Route ---
# --- Route for Text Block Generation ---
# --- Route for Text Block Generation ---
@app.route('/generate_text_block', methods=['POST'])
def generate_text_block_route():
    print("Received request at /generate_text_block")

    # --- Check if client object exists ---
    if client is None:
        print("Error condition met: client object is None before processing request.")
        # This specific message confirms the init failed earlier
        return jsonify({'status': 'error', 'message': 'Server setup error: AI client object is None.'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        topic = data.get('topic')
        grade_level = data.get('grade_level', 'middle school')
        focus = data.get('focus')

        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic" for text generation'}), 400
        print(f"Received Text Block request: Topic='{topic}', Grade='{grade_level}', Focus='{focus}'")

        # --- Create Prompt ---
        prompt_content = create_text_block_prompt(topic, grade_level, focus)

        # --- Call Anthropic API ---
        print("Attempting to call Anthropic API...") # Log before call
        # *** Add specific Anthropic error catches below ***
        try:
            message = client.messages.create(
                model=ANTHROPIC_MODEL_NAME,
                max_tokens=800,
                temperature=0.7,
                messages=[{ "role": "user", "content": prompt_content }]
            )
            print("Received response from Anthropic API successfully.") # Log after successful call
        except anthropic.AuthenticationError as auth_err:
             print(f"Anthropic Authentication Error: {auth_err}")
             # Return a specific error for bad keys
             return jsonify({'status': 'error', 'message': f'AI Authentication Error: {auth_err}'}), 401 # Unauthorized
        except anthropic.APIConnectionError as conn_err:
             print(f"Anthropic Connection Error: {conn_err}")
             return jsonify({'status': 'error', 'message': f'AI Connection Error: {conn_err}'}), 503 # Service Unavailable
        except anthropic.RateLimitError as rate_err:
             print(f"Anthropic Rate Limit Error: {rate_err}")
             return jsonify({'status': 'error', 'message': 'AI Rate Limit Exceeded.'}), 429
        except anthropic.APIStatusError as status_err:
            # --- Start of APIStatusError block ---
            print(f"Anthropic API Status Error: Status Code: {status_err.status_code}, Response: {status_err.response}")
            error_message = f'AI service error (Status {status_err.status_code})' # Semicolon removed

            # --- Start of nested try (indented) ---
            try:
                error_details = status_err.response.json() # Semicolon removed
                error_message += f": {error_details.get('error', {}).get('message', status_err.response.text)}" # Semicolon removed
            # --- Nested except (indented to match nested try) ---
            except Exception:
                # --- Line inside nested except (indented further) ---
                error_message += f": {status_err.response.text}" # Semicolon removed
            # --- End of nested try...except ---

            # This return belongs to the outer APIStatusError block (indented same as print/error_message assignment)
            return jsonify({'status': 'error', 'message': error_message}), status_err.status_code
        # --- End of APIStatusError block ---

        except Exception as api_call_err: # Correctly indented relative to outer 
             
             print(f"Unexpected error DURING Anthropic API call: {api_call_err}")
             print(traceback.format_exc())
             return jsonify({'status': 'error', 'message': f'Unexpected error during AI call: {api_call_err}'}), 500

        # --- Extract Result (only if API call succeeded) ---
        generated_content = ""
        if message and message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            generated_content = message.content[0].text
        else:
             print(f"Warning: Unexpected API response structure or empty content after successful call. Response: {message}"); return jsonify({'status': 'error', 'message': 'Failed to parse content from AI response.'}), 500

        print(f"Generated Text Block length: {len(generated_content)} chars")

        # --- Return Result ---
        return jsonify({
            'status': 'success',
            'text_block_content': generated_content.strip()
        })

    # --- Catch errors outside the API call itself ---
    except ValueError as ve: print(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e: # General errors in the route logic (e.g., getting JSON data)
        print(f"An unexpected error occurred in /generate_text_block route logic: {e}"); print(traceback.format_exc()); return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500
    except anthropic.APIConnectionError as e: print(f"API Connection Error: {e}"); return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e: print(f"API Rate Limit Error: {e}"); return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    except anthropic.APIStatusError as e:
            print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
            error_message = f'AI service error (Status {e.status_code})'
            # --- Start of nested try ---
            try:
                # Try to parse more specific error details from the JSON response body
                error_details = e.response.json()
                error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
            # --- Nested except - correctly indented ---
            except Exception:
                # --- Line inside nested except - correctly indented ---
                # Fallback to using the raw response text if JSON parsing fails
                error_message += f": {e.response.text}"
            # --- End of nested try...except ---

            # This return statement is correctly indented for the APIStatusError block
            return jsonify({'status': 'error', 'message': error_message}), e.status_code

        # --- This except block starts on a NEW LINE and is indented correctly ---
    except Exception as e:
            # --- Lines inside this except block are correctly indented ---
        print(f"An unexpected error occurred in /generate_text_block: {e}") # Ensure route name is correct
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500
# --- End of Text Block Route ---
# --- Route for Short Answer Question Generation ---
@app.route('/generate_short_answer', methods=['POST'])
def generate_short_answer_route():
    """Handles POST requests to generate Short Answer questions and keys."""
    print("Received request at /generate_short_answer")
    # ... (Keep client check, json check) ...
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        topic = data.get('topic')
        grade_level = data.get('grade_level', 'middle school')
        num_questions_req = data.get('num_questions', 5)

        # --- Validation (Keep existing validation) ---
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic" in request data'}), 400
        try:
            num_questions = int(num_questions_req);
            if not 2 <= num_questions <= 10: raise ValueError("Number of questions must be between 2 and 10.")
        except (ValueError, TypeError): return jsonify({'status': 'error', 'message': 'Invalid number of questions specified (must be an integer between 2 and 10).'}), 400

        print(f"Received SA request: Topic='{topic}', Grade='{grade_level}', NumQuestions={num_questions}")

        # --- Create Prompt (Uses the *updated* prompt function) ---
        prompt_content = create_short_answer_prompt(topic, grade_level, num_questions)

        # --- Call Anthropic API ---
        print(f"Sending SA request to Anthropic API (Model: {ANTHROPIC_MODEL_NAME})...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=1200 + (num_questions * 100), # Increased slightly for answers
            temperature=0.7,
            messages=[{ "role": "user", "content": prompt_content }]
        )
        print("Received response from Anthropic API.")

        # --- Extract Result (No change needed here) ---
        generated_content = ""
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            generated_content = message.content[0].text
        else:
             print(f"Warning: Unexpected API response structure or empty content. Response: {message}"); return jsonify({'status': 'error', 'message': 'Failed to parse content from AI response.'}), 500

        print(f"Generated SA content length: {len(generated_content)} chars")

        # --- Return Result (No change needed here) ---
        return jsonify({
            'status': 'success',
            'short_answer_content': generated_content.strip() # Key 'short_answer_content' now includes Qs and Key
        })

    # --- Error Handling (Keep existing handlers) ---
    # ... except ValueError ...
    # ... except anthropic errors ...
    # ... except Exception ...
        # --- Error Handling ---
    except ValueError as ve:
         print(f"Validation Error: {ve}")
         return jsonify({'status': 'error', 'message': str(ve)}), 400
    except anthropic.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e:
        print(f"API Rate Limit Error: {e}")
        return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    # *** CORRECTED BLOCK BELOW ***
    except anthropic.APIStatusError as e:
        print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
        error_message = f'AI service error (Status {e.status_code})'
        try:
            error_details = e.response.json()
            error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception: # Catch potential JSON decoding errors or different structures
            error_message += f": {e.response.text}" # Fallback to raw text
        return jsonify({'status': 'error', 'message': error_message}), e.status_code
    # *** END OF CORRECTED BLOCK ***
    except Exception as e:
        print(f"An unexpected error occurred in /generate_short_answer: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'An internal server error occurred.'}), 500

# --- End of Short Answer Route ---
# --- Route to Serve the Frontend HTML ---
@app.route('/')
def serve_index():
    """Serves the index.html file."""
    print("Serving index.html")
    return send_from_directory('.', 'index.html')
# --- Route to Save Generated/Edited Item ---
@app.route('/save_item', methods=['POST'])
def save_item_route():
    """Handles POST requests to save an item to the database."""
    print("Received request at /save_item")

    if not request.is_json:
        print("Error: Request is not JSON for /save_item")
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    data = request.get_json()

    # --- Extract data from payload ---
    item_type = data.get('item_type')
    source_topic = data.get('source_topic')
    source_url = data.get('source_url')
    grade_level = data.get('grade_level')
    content_html = data.get('content_html')

    # --- Basic Validation ---
    if not item_type or not grade_level or not content_html:
         return jsonify({'status': 'error', 'message': 'Missing required fields: item_type, grade_level, content_html'}), 400
    if item_type == 'gapFill' and not source_topic:
         return jsonify({'status': 'error', 'message': 'Missing source_topic for gapFill item'}), 400
    if item_type == 'youtube' and not source_url:
         return jsonify({'status': 'error', 'message': 'Missing source_url for youtube item'}), 400

    print(f"Attempting to save item: type={item_type}, topic={source_topic}, url={source_url}, grade={grade_level}, html_len={len(content_html or '')}")

    try:
        # --- Create Database Record ---
        new_item = GeneratedItem(
            item_type=item_type,
            source_topic=source_topic,
            source_url=source_url,
            grade_level=grade_level,
            content_html=content_html
            # Timestamps are handled by default/onupdate
        )

        # --- Add to session and commit ---
        db.session.add(new_item)
        db.session.commit()

        print(f"Successfully saved item with ID: {new_item.id}")
        return jsonify({'status': 'success', 'message': 'Item saved successfully!', 'item_id': new_item.id})

    except Exception as e:
        db.session.rollback() # Rollback transaction on error
        print(f"Error saving item to database: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f'Database error: {e}'}), 500

# --- End of Save Route ---
# --- Route to Get Full Content of a Specific Saved Item ---
@app.route('/get_item/<int:item_id>', methods=['GET'])
def get_item_route(item_id):
    """Handles GET requests to retrieve full details for a specific item ID."""
    print(f"Received request at /get_item/{item_id}")
    try:
        # Query the database for the item by its primary key (ID)
        # .get_or_404() is convenient: returns item or aborts with 404 if not found
        item = db.session.get(GeneratedItem, item_id) # Simpler query for primary key

        if item is None:
            print(f"Item with ID {item_id} not found.")
            return jsonify({'status': 'error', 'message': 'Item not found in library.'}), 404

        print(f"Found item: {item.item_type}, Topic: {item.source_topic}, URL: {item.source_url}")

        # Return the full details, including the crucial content_html
        item_data = {
            'id': item.id,
            'item_type': item.item_type,
            'source_topic': item.source_topic,
            'source_url': item.source_url,
            'grade_level': item.grade_level,
            'content_html': item.content_html, # Send the full HTML content
            'creation_date': item.creation_date.isoformat(),
            'last_modified_date': item.last_modified_date.isoformat()
        }
        return jsonify({'status': 'success', 'item': item_data})

    except Exception as e:
        print(f"Error retrieving item {item_id} from database: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': 'Error retrieving item details.'}), 500

# --- End of Get Item Route ---
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
        data = request.get_json()
        topic = data.get('topic')
        grade_level = data.get('grade_level', 'middle school') # Default if missing

        # Log received data
        print(f"Received data: {data}")

        if not topic:
            print("Error: 'topic' missing in request data.")
            return jsonify({'status': 'error', 'message': 'Missing "topic" in request data'}), 400

        # Log the topic and grade level correctly
        print(f"Received topic: '{topic}', Grade Level: '{grade_level}'")

        # Create the prompt
        prompt_content = create_gap_fill_prompt(topic, grade_level)

        # --- Call Anthropic API ---
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

        # Extract the generated text
        generated_text = ""
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            generated_text = message.content[0].text
        else:
            print(f"Warning: Unexpected API response structure or empty content. Response: {message}")
            return jsonify({'status': 'error', 'message': 'Failed to parse content from API response.'}), 500

        # Log content length
        print(f"Generated content length: {len(generated_text)} characters")

        # --- Return the Result to the Frontend ---
        return jsonify({
            'status': 'success',
            'worksheet_content': generated_text.strip()
        })

    # --- Specific Error Handling for Anthropic API ---
    except anthropic.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e:
        print(f"API Rate Limit Error: {e}")
        return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    except anthropic.APIStatusError as e:
        print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
        error_message = f'AI service error (Status {e.status_code})'
        try:
            error_details = e.response.json()
            error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception:
            error_message += f": {e.response.text}"
        return jsonify({'status': 'error', 'message': error_message}), e.status_code
    # --- General Error Handling ---
    except Exception as e:
        print(f"An unexpected error occurred in /generate_worksheet: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f'An internal server error occurred.'}), 500

@app.route('/generate_comprehension', methods=['POST'])
def generate_comprehension_route():
    """Handles POST requests to generate comprehension questions from a YouTube URL."""
    print("Received request at /generate_comprehension")

    if client is None:
        print("Error: Anthropic client not initialized.")
        return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500

    if not request.is_json:
        print("Error: Request is not JSON.")
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    try:
        data = request.get_json()
        youtube_url = data.get('youtube_url')
        num_questions_req = data.get('num_questions', 5)  # Get value, default to 5
        
        try:
            num_questions = int(num_questions_req)
            if not 2 <= num_questions <= 20:  # Validate range (e.g., 2-20)
                print(f"Error: Invalid number of questions requested: {num_questions}")
                raise ValueError("Number of questions must be between 2 and 20.")
        except (ValueError, TypeError):
            print(f"Error: Invalid type or value for num_questions: {num_questions_req}")
            return jsonify({'status': 'error', 'message': 'Invalid number of questions specified (must be an integer between 2 and 20).'}), 400

        if not youtube_url:
            return jsonify({'status': 'error', 'message': 'Missing "youtube_url" in request data'}), 400

        # Log the number of questions requested
        print(f"Received YouTube URL: {youtube_url}, Num Questions: {num_questions}")

        # --- Extract Video ID ---
        video_id = extract_video_id(youtube_url)
        if not video_id:
            print(f"Error: Could not extract Video ID from URL: {youtube_url}")
            return jsonify({'status': 'error', 'message': f'Invalid YouTube URL format: {youtube_url}'}), 400
        
        print(f"Extracted Video ID: {video_id}")

        # --- Get Transcript ---
        transcript_text = ""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US'])  # Prioritize English
            # Combine transcript parts into a single string
            transcript_text = " ".join([item['text'] for item in transcript_list])
            print(f"Successfully fetched transcript, length: {len(transcript_text)} characters.")
            if not transcript_text.strip():  # Check if transcript is empty after joining
                print(f"Warning: Fetched transcript for {video_id} is empty.")
                return jsonify({'status': 'error', 'message': 'Transcript found but it is empty.'}), 400
                
        except TranscriptsDisabled:
            print(f"Error: Transcripts are disabled for video: {video_id}")
            return jsonify({'status': 'error', 'message': 'Transcripts are disabled for this video.'}), 400
        except NoTranscriptFound:
            print(f"Error: No English transcript found for video: {video_id}")
            return jsonify({'status': 'error', 'message': 'No English transcript found for this video.'}), 404
        except Exception as e:  # Catch other potential errors from the library
            print(f"Error fetching transcript for {video_id}: {e}")
            print(traceback.format_exc())
            return jsonify({'status': 'error', 'message': f'Could not fetch transcript: {e}'}), 500

        # --- Generate Comprehension Questions Prompt ---
        # Pass the validated num_questions to the prompt function
        prompt_content = create_comprehension_prompt(transcript_text, num_questions)

        # --- Call Anthropic API ---
        print(f"Sending transcript to Anthropic API (Model: {ANTHROPIC_MODEL_NAME}, requesting {num_questions} questions)...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=1000 + (num_questions * 50),  # Dynamically adjust tokens slightly based on question count
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        )
        print("Received response from Anthropic API.")

        # --- Extract Result ---
        generated_questions = ""
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'):
            generated_questions = message.content[0].text
        else:
            print(f"Warning: Unexpected API response structure or empty content. Response: {message}")
            return jsonify({'status': 'error', 'message': 'Failed to parse questions from AI response.'}), 500

        print(f"Generated Questions length: {len(generated_questions)} chars")

        # --- Return Result ---
        return jsonify({
            'status': 'success',
            'comprehension_questions': generated_questions.strip()
        })

    # --- Error Handling ---
    except anthropic.APIConnectionError as e:
        print(f"API Connection Error: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to connect to AI service: {e}'}), 503
    except anthropic.RateLimitError as e:
        print(f"API Rate Limit Error: {e}")
        return jsonify({'status': 'error', 'message': 'Rate limit exceeded. Please try again later.'}), 429
    except anthropic.APIStatusError as e:
        print(f"API Status Error: Status Code: {e.status_code}, Response: {e.response}")
        error_message = f'AI service error (Status {e.status_code})'
        try:
            error_details = e.response.json()
            error_message += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception:
            error_message += f": {e.response.text}"
        return jsonify({'status': 'error', 'message': error_message}), e.status_code
    except ValueError as ve:  # Catch the validation error we added
        print(f"Validation Error: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400  # Return 400 for validation errors
    except Exception as e:
        print(f"An unexpected error occurred in /generate_comprehension: {e}")
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f'An internal server error occurred.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)