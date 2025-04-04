import os
import anthropic
# Near other imports
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone # Added timezone for UTC awareness
import os # Make sure os is imported if not already
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import traceback # Import traceback for detailed error logging
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
    app.run(host='127.0.0.1', port=5001, debug=True)