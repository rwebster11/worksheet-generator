import os
import io
import re # For parsing YouTube URL
import random # Needed for word search logic
from copy import deepcopy # Needed for word search logic
import traceback # Import traceback for detailed error logging
import logging # Import standard logging
from datetime import datetime, timezone # Added timezone for UTC awareness
from PIL import Image, ImageDraw, ImageFont
import math
# Third-party imports
import anthropic
import httpx # Still needed by Anthropic client internally, even without proxy
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from bs4 import BeautifulSoup
import docx
from docx.shared import Inches, Pt
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound # Keep for now

# Configure basic logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

# Load environment variables from .env file
logging.info("Attempting to load .env file...")
dotenv_path = os.path.join(os.path.dirname(__file__), '.env') # Explicit path
found_dotenv = load_dotenv(dotenv_path=dotenv_path, verbose=True) # Be verbose
logging.info(f".env file found and loaded: {found_dotenv}")
# Removed duplicate load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# --- Database Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'worksheet_library.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# --- End Database Configuration ---

# Initialize the SQLAlchemy extension
db = SQLAlchemy(app)

# --- Database Model Definitions ---

# Association Table: Links Worksheets and GeneratedItems, storing order
worksheet_items_association = db.Table('worksheet_items_association',
    db.Column('worksheet_id', db.Integer, db.ForeignKey('worksheets.id'), primary_key=True),
    db.Column('generated_item_id', db.Integer, db.ForeignKey('generated_items.id'), primary_key=True), # Matches your column name
    db.Column('item_order', db.Integer, nullable=False)
)

# GeneratedItem Model
class GeneratedItem(db.Model):
    __tablename__ = 'generated_items'
    id = db.Column(db.Integer, primary_key=True)
    item_type = db.Column(db.String(50), nullable=False)
    source_topic = db.Column(db.String(250), nullable=True)
    source_url = db.Column(db.String(500), nullable=True)
    grade_level = db.Column(db.String(50), nullable=False)
    content_html = db.Column(db.Text, nullable=False)
    creation_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_modified_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    worksheets = db.relationship(
        'Worksheet',
        secondary='worksheet_items_association',
        back_populates='items',
        lazy='selectin'
    )

    def __repr__(self):
        return f'<GeneratedItem id={self.id} type={self.item_type}>'

# Worksheet Model
class Worksheet(db.Model):
    __tablename__ = 'worksheets'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(250), nullable=False, default="Untitled Worksheet")
    creation_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    last_modified_date = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    items = db.relationship(
        'GeneratedItem',
        secondary='worksheet_items_association',
        order_by=worksheet_items_association.c.item_order,
        back_populates='worksheets',
        lazy='selectin'
    )

    def __repr__(self):
        return f'<Worksheet id={self.id} title="{self.title}">'

# Configure mappers after models are defined
db.configure_mappers()

# --- Anthropic Client Initialization ---
client = None
try:
    logging.info("Attempting to initialize Anthropic client...")
    api_key_from_env = os.getenv("ANTHROPIC_API_KEY")
    if not api_key_from_env:
        logging.error("CRITICAL: ANTHROPIC_API_KEY not found in environment variables.")
    else:
        logging.info("ANTHROPIC_API_KEY found. Initializing client...")
        # Initialize without unused proxy setup
        client = anthropic.Anthropic(api_key=api_key_from_env)
        logging.info("Anthropic client object CREATED successfully.")
except Exception as e:
    logging.error(f"CRITICAL: Exception during Anthropic client initialization: {e}", exc_info=True)
    client = None

if client is None:
    logging.warning("Anthropic client is None after initialization attempt.")
else:
    logging.info("Anthropic client appears to be initialized.")

ANTHROPIC_MODEL_NAME = "claude-3-opus-20240229"

# --- Prompt Creation Functions ---
def create_mcq_prompt(topic, grade_level, num_questions=5):
    # ... (keep function as is) ...
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

def create_word_list_prompt(topic, grade_level, num_words=15):
    # ... (keep function as is) ...
    prompt = f"""You are a helpful vocabulary assistant.
Generate a list of exactly {num_words} single keywords or very short (2-word max) key phrases relevant to the topic '{topic}' for a {grade_level} audience.

**Instructions:**
1.  Focus on specific nouns, verbs, or essential terms related to '{topic}'.
2.  Ensure words are appropriate for the {grade_level}.
3.  Provide *only* the list of words/phrases.
4.  Format the output as a numbered list (e.g., "1. Photosynthesis", "2. Chlorophyll", "3. Carbon Dioxide").
5.  Do not include definitions, explanations, or any text other than the numbered list.

Topic: {topic}
Audience: {grade_level}
Number of Words: {num_words}

Keyword List:
"""
    return prompt

def create_text_block_prompt(topic, grade_level, focus=None):
    # ... (keep function as is) ...
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
    # ... (keep function as is) ...
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

def create_gap_fill_prompt(topic, grade_level="middle school", num_sentences=7):
    # ... (keep function as is) ...
    prompt = f"""You are an expert teacher creating educational resources.
Generate a gap-fill (cloze) activity about the topic '{topic}'.
The target audience is {grade_level} students.

**Instructions:**
1.  Create exactly {num_sentences} unique sentences. Each sentence **MUST** focus on a **different** important aspect or key term of the topic '{topic}'. Do not just rephrase the same core idea.
2.  In each sentence, identify the single most important key term or concept specific to that sentence's point and replace it with '_________'.
3.  **CRITICAL REQUIREMENT: Each blank MUST be fillable with a DIFFERENT word.** Do NOT reuse the same answer word for multiple blanks. The goal is to test a variety of key vocabulary related to the topic. Using the exact same word in two or more blanks is forbidden.
4.  After the sentences, provide a numbered list titled 'Answer Key:' that clearly lists the single word removed for each blank in the correct order.
5.  Output ONLY the worksheet sentences and the Answer Key. No introductory phrases, explanations, conversational text, or titles other than 'Answer Key:'.

Topic: {topic}
Audience: {grade_level}
Number of Sentences: {num_sentences}

Worksheet:
"""
    return prompt

def extract_video_id(url): # Keep for now
    # ... (keep function as is) ...
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None

def create_comprehension_prompt(transcript_text, num_questions=5): # Keep for now
    # ... (keep function as is) ...
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

def create_pasted_comprehension_prompt(pasted_text, num_questions=5):
    # ... (keep function as is) ...
    max_chars = 15000
    truncated_text = pasted_text[:max_chars]
    if len(pasted_text) > max_chars: print(f"Warning: Pasted text truncated to {max_chars} chars for AI prompt.")
    prompt = f"""You are an expert educator designing reading comprehension assessments.
Based *only* on the following provided text passage, generate {num_questions} insightful comprehension questions that test understanding of the key information presented within that text.
Also provide a brief answer key outlining the expected points or a model answer for each question based *only* on the provided text.

**CRITICAL INSTRUCTIONS:**
1.  **Question Generation:** Create {num_questions} distinct questions directly related to the content of the provided text passage. Questions should require recall, explanation, or analysis *of the passage*.
2.  **Question Formatting:** Present each question clearly, starting with a number (e.g., "1.", "2.").
3.  **Answer Key Generation:** After ALL questions are listed, provide a separate section titled exactly "Answer Key:".
4.  **Key Content:** For each question number in the Answer Key section, provide key points or a model answer derived *solely from the provided passage*.
5.  **Key Formatting:** Start each answer key item with the corresponding question number (e.g., "1.", "2.").
6.  **Strict Output:** Output ONLY the numbered list of questions, followed by the "Answer Key:" section and the key itself. No extra text, introductions, or commentary about the text quality.

**Provided Text Passage:**
--- START TEXT ---
{truncated_text}
--- END TEXT ---

**{num_questions} Comprehension Questions:**
[Generate Questions Here Following Format]

Answer Key:
[Generate Answer Key Here Following Format]
"""
    return prompt

def create_short_answer_prompt(topic, grade_level, num_questions=5):
    # ... (keep function as is) ...
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

def create_similar_questions_prompt(example_question, num_questions=3, grade_level="Not Specified"):
    # ... (keep function as is) ...
    prompt = f"""
You are an expert question designer replicating educational assessment items. Your task is to analyze the provided example question, identify the core concept and calculation type being tested, and then generate {num_questions} NEW questions that assess the SAME core concept at a SIMILAR difficulty level suitable for {grade_level} students.

**Analysis of Example:**
First, briefly state the core concept or skill tested by the example question. (e.g., "Calculating speed given distance and time", "Solving a two-step linear equation", "Calculating molarity").

**Instructions for New Questions:**
1.  **Maintain Core Concept:** Each new question MUST test the exact same fundamental principle or calculation type as the example.
2.  **Maintain Difficulty:** The complexity of the numbers used, the number of steps required, and the required conceptual understanding should closely match the example question and be appropriate for the target {grade_level}.
3.  **Construct Realistic Scenarios:** For each new question:
    a.  **Determine a realistic target answer** appropriate for the concept and grade level (e.g., a percentage yield between 60-95%, a reasonable speed for a car).
    b.  **Work Backwards:** Choose plausible input numbers (like initial mass, theoretical yield, distance, time, etc.) that will mathematically lead to the target answer you determined. Ensure these input numbers are also contextually sensible (e.g., realistic masses, times, distances).
    c.  **Vary Context & Specifics:** Create a genuinely new scenario (different chemicals, objects, situations) using these generated numbers. You can vary units where appropriate (e.g., grams to kg) and potentially the specific variable being solved for (if applicable), while keeping the underlying problem type identical to the example. Do NOT just rephrase the example question.
4.  **Formatting:** Present each new question clearly, starting with a number (e.g., "1.", "2."). List each question on a new line.
5.  **Answer Key Generation:** After ALL new questions are listed, provide a separate section titled exactly "Answer Key:".
6. Key Content: In the Answer Key section, for each question number, show the key calculation steps required to solve the problem, followed by the final numerical answer (including units if appropriate). Make the steps clear and concise.

    Example Format:
    1. Steps:
       Step 1: Formula (e.g., Speed = Distance / Time)
       Step 2: Substitute values (e.g., Speed = 300 miles / 5 hours)
       Step 3: Calculate (e.g., Speed = 60)
       Final Answer: 60 mph
    2. Steps:
       ...
       Final Answer: ...

7.  **Strict Output:** Output ONLY the brief "Analysis of Example", the numbered list of new questions, the "Answer Key:" title, and the key itself. No extra text, introductions, or explanations.

**Example Question Provided:**
--- START EXAMPLE ---
{example_question}
--- END EXAMPLE ---

**Analysis of Example:**
[AI writes analysis here]

**{num_questions} New Similar Questions:**
[AI generates new questions here]

Answer Key:
[AI generates answer key for the NEW questions here]
"""
    return prompt

# --- Word Search Generation Logic (Adapted from make_wordsearch.py) ---

# Constants
NMAX_GRID = 32
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# --- Masking Functions ---
def circle_mask(grid, nrows, ncols):
    r2 = min(ncols, nrows)**2 // 4
    cx, cy = ncols//2, nrows // 2
    for irow in range(nrows):
        for icol in range(ncols):
            if (irow - cy)**2 + (icol - cx)**2 > r2:
                grid[irow][icol] = '*'
def squares_mask(grid, nrows, ncols):
    a = int(0.38 * min(ncols, nrows))
    cy = nrows // 2; cx = ncols // 2
    for irow in range(nrows):
        for icol in range(ncols):
            if a <= icol < ncols-a:
                if irow < cy-a or irow > cy+a: grid[irow][icol] = '*'
            if a <= irow < nrows-a:
                if icol < cx-a or icol > cx+a: grid[irow][icol] = '*'
def no_mask(grid, nrows, ncols): pass
apply_mask = {None: no_mask, 'circle': circle_mask, 'squares': squares_mask}

# --- Grid Creation and Filling ---
def make_initial_grid(nrows, ncols, mask_name=None):
    if mask_name not in apply_mask:
        logging.warning(f"Unknown mask name '{mask_name}'. Using no mask.")
        mask_name = None
    grid = [[' ']*ncols for _ in range(nrows)]
    apply_mask[mask_name](grid, nrows, ncols)
    return grid

def fill_grid_randomly(grid, nrows, ncols):
    for irow in range(nrows):
        for icol in range(ncols):
            if grid[irow][icol] == ' ': grid[irow][icol] = random.choice(ALPHABET)

def remove_mask_chars(grid, nrows, ncols):
    for irow in range(nrows):
        for icol in range(ncols):
            if grid[irow][icol] == '*': grid[irow][icol] = ' '

# --- Word Placement Logic ---
def _try_place_words(nrows, ncols, wordlist, allow_backwards_words=True, mask_name=None):
    grid = make_initial_grid(nrows, ncols, mask_name)

    def test_candidate(grid, irow, icol, dx, dy, word):
        word_len = len(word)
        for j in range(word_len):
            test_row, test_col = irow + j*dy, icol + j*dx
            if not (0 <= test_row < nrows and 0 <= test_col < ncols): return False
            if grid[test_row][test_col] not in (' ', word[j]): return False
        return True

    def place_word_in_grid(grid, word):
        word = ''.join(filter(str.isalnum, word)).upper()
        if not word: return None
        dxdy_choices = [(0,1), (1,0), (1,1), (1,-1)]; random.shuffle(dxdy_choices)

        for (dx, dy) in dxdy_choices:
            word_to_place = word
            if allow_backwards_words and random.choice([True, False]): word_to_place = word[::-1]

            n = len(word_to_place)
            colmin = 0; colmax = ncols - n if dx != 0 else ncols - 1
            rowmin = 0 if dy >= 0 else n - 1; rowmax = nrows - n if dy > 0 else nrows - 1
            if dy < 0: rowmax = nrows -1 # Correct range for up-diagonals

            if colmax < colmin or rowmax < rowmin: continue

            candidates = []
            for r in range(rowmin, rowmax + 1):
                 for c in range(colmin, colmax + 1):
                      end_r, end_c = r + (n-1)*dy, c + (n-1)*dx
                      if 0 <= end_r < nrows and 0 <= end_c < ncols:
                           if test_candidate(grid, r, c, dx, dy, word_to_place): candidates.append((r, c))

            if not candidates: continue
            start_row, start_col = random.choice(candidates)
            current_row, current_col = start_row, start_col
            for char in word_to_place:
                grid[current_row][current_col] = char
                current_row += dy; current_col += dx
            logging.debug(f"Placed '{word}' at ({start_row},{start_col}) orientation ({dx},{dy})")
            return True # Indicate success

        logging.warning(f"Failed to place word: {word}")
        return False

    successfully_placed_words = []
    sorted_wordlist = sorted([w for w in wordlist if w], key=len, reverse=True)
    for word in sorted_wordlist:
        if place_word_in_grid(grid, word): successfully_placed_words.append(word.upper())
        else: return None, None

    solution_grid = deepcopy(grid); fill_grid_randomly(grid, nrows, ncols)
    remove_mask_chars(grid, nrows, ncols); remove_mask_chars(solution_grid, nrows, ncols)
    return grid, sorted(successfully_placed_words)

# --- Top-Level Generation Function ---
class PuzzleGenerationError(Exception): pass

def generate_wordsearch_grid(wordlist, nrows, ncols, allow_backwards_words=True, mask_name=None, attempts=10):
    if nrows > NMAX_GRID or ncols > NMAX_GRID: raise ValueError(f'Max grid dimension is {NMAX_GRID}')
    if not wordlist: raise ValueError("Word list cannot be empty.")

    max_dimension = max(nrows, ncols); cleaned_wordlist = []
    for word in wordlist:
         cleaned_word = ''.join(filter(str.isalnum, word)).upper()
         if cleaned_word:
              if len(cleaned_word) > max_dimension: raise ValueError(f"Word '{cleaned_word}' > {max_dimension} chars (grid {nrows}x{ncols})")
              cleaned_wordlist.append(cleaned_word)
    if not cleaned_wordlist: raise ValueError("No valid words after cleaning.")

    for i in range(attempts):
        logging.info(f"Word search generation attempt {i+1}/{attempts}...")
        grid, placed_words = _try_place_words(nrows, ncols, cleaned_wordlist, allow_backwards_words, mask_name)
        if grid is not None:
            logging.info(f"Successfully generated grid in {i+1} attempt(s). Placed {len(placed_words)} words.")
            return grid, placed_words
    raise PuzzleGenerationError(f"Failed to place all words after {attempts} attempts.")
# --- End of Word Search Generation Logic ---
def create_wordsearch_image(grid_list, cell_size=30, font_size=18, font_path=None):
    """Creates a PIL Image object of the word search grid."""
    if not grid_list or not grid_list[0]:
        return None # Cannot generate image from empty grid

    nrows = len(grid_list)
    ncols = len(grid_list[0])
    img_width = ncols * cell_size
    img_height = nrows * cell_size

    # Create white background image
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    # Try to load a font (adjust path if needed, or handle fallback)
    try:
        # On Linux servers, common paths might be different.
        # A basic sans-serif font might be more reliable if Century Gothic isn't installed server-side.
        # Example paths (check your system):
        # font_path_try = font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" # Common on Linux
        font_path_try = font_path or "arial.ttf" # Common on Windows, might work locally
        font = ImageFont.truetype(font_path_try, font_size)
    except IOError:
        logging.warning(f"Font file not found at {font_path_try}. Using default PIL font.")
        font = ImageFont.load_default() # Fallback font

    # Draw letters and grid lines
    for r in range(nrows):
        for c in range(ncols):
            char = grid_list[r][c]
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            # Draw cell borders
            draw.rectangle([x0, y0, x1, y1], outline='grey')

            # Draw letter centered in cell
            # text_width, text_height = draw.textsize(char, font=font) # Deprecated
            bbox = draw.textbbox((0,0), char, font=font) # x0, y0, x1, y1
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x0 + (cell_size - text_width) / 2 - bbox[0] # Adjust for bbox offset
            text_y = y0 + (cell_size - text_height) / 2 - bbox[1] # Adjust for bbox offset
            draw.text((text_x, text_y), char, fill='black', font=font)

    return image
# --- Standard API Routes ---
# (Generic Error Handlers - Can be refactored later)
def handle_anthropic_error(e):
    if isinstance(e, anthropic.APIConnectionError): code=503; msg=f'AI Connection Error: {e}'
    elif isinstance(e, anthropic.RateLimitError): code=429; msg='AI Rate Limit Exceeded.'
    elif isinstance(e, anthropic.AuthenticationError): code=401; msg=f'AI Authentication Error: {e}'
    elif isinstance(e, anthropic.APIStatusError):
        code = e.status_code; msg = f'AI service error (Status {code})'
        try: error_details = e.response.json(); msg += f": {error_details.get('error', {}).get('message', e.response.text)}"
        except Exception: msg += f": {e.response.text}"
    else: code=500; msg='An unexpected error occurred during AI call.'
    logging.error(f"{msg} - Exception: {e}", exc_info=(code == 500)) # Log traceback for unexpected
    return jsonify({'status': 'error', 'message': msg}), code

# --- List Saved Items Route ---
@app.route('/list_items', methods=['GET'])
def list_items_route():
    logging.info("Received request at /list_items")
    try:
        items = GeneratedItem.query.order_by(GeneratedItem.last_modified_date.desc()).limit(20).all()
        items_list = [{'id': item.id, 'item_type': item.item_type, 'source_topic': item.source_topic,
                       'source_url': item.source_url, 'grade_level': item.grade_level,
                       'last_modified': item.last_modified_date.isoformat()} for item in items]
        return jsonify({'status': 'success', 'items': items_list})
    except Exception as e:
        logging.error(f"Error retrieving items from database: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Error retrieving items from library.'}), 500

# --- Generate Routes ---
@app.route('/generate_true_false', methods=['POST'])
def generate_true_false_route():
    logging.info("Received request at /generate_true_false")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); topic = data.get('topic'); grade_level = data.get('grade_level', 'middle school'); num_statements_req = data.get('num_statements', 8)
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic"'}), 400
        try: num_statements = int(num_statements_req); assert 3 <= num_statements <= 20
        except (ValueError, TypeError, AssertionError): return jsonify({'status': 'error', 'message': 'Invalid num_statements (must be int 3-20)'}), 400
        logging.info(f"T/F request: Topic='{topic}', Grade='{grade_level}', NumStatements={num_statements}")
        prompt_content = create_true_false_prompt(topic, grade_level, num_statements)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=1000 + (num_statements * 50), temperature=0.6, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated T/F content length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'true_false_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_true_false: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_mcq', methods=['POST'])
def generate_mcq_route():
    logging.info("Received request at /generate_mcq")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); topic = data.get('topic'); grade_level = data.get('grade_level', 'middle school'); num_questions_req = data.get('num_questions', 5)
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic"'}), 400
        try: num_questions = int(num_questions_req); assert 2 <= num_questions <= 15
        except (ValueError, TypeError, AssertionError): return jsonify({'status': 'error', 'message': 'Invalid num_questions (must be int 2-15)'}), 400
        logging.info(f"MCQ request: Topic='{topic}', Grade='{grade_level}', NumQuestions={num_questions}")
        prompt_content = create_mcq_prompt(topic, grade_level, num_questions)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=1500 + (num_questions * 100), temperature=0.7, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated MCQ content length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'mcq_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_mcq: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_text_block', methods=['POST'])
def generate_text_block_route():
    logging.info("Received request at /generate_text_block")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); topic = data.get('topic'); grade_level = data.get('grade_level', 'middle school'); focus = data.get('focus')
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic"'}), 400
        logging.info(f"Text Block request: Topic='{topic}', Grade='{grade_level}', Focus='{focus}'")
        prompt_content = create_text_block_prompt(topic, grade_level, focus)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=800, temperature=0.7, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated Text Block length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'text_block_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_text_block: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_short_answer', methods=['POST'])
def generate_short_answer_route():
    logging.info("Received request at /generate_short_answer")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); topic = data.get('topic'); grade_level = data.get('grade_level', 'middle school'); num_questions_req = data.get('num_questions', 5)
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic"'}), 400
        try: num_questions = int(num_questions_req); assert 2 <= num_questions <= 10
        except (ValueError, TypeError, AssertionError): return jsonify({'status': 'error', 'message': 'Invalid num_questions (must be int 2-10)'}), 400
        logging.info(f"SA request: Topic='{topic}', Grade='{grade_level}', NumQuestions={num_questions}")
        prompt_content = create_short_answer_prompt(topic, grade_level, num_questions)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=1200 + (num_questions * 100), temperature=0.7, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated SA content length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'short_answer_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_short_answer: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_pasted_comprehension', methods=['POST'])
def generate_pasted_comprehension_route():
    logging.info("Received request at /generate_pasted_comprehension")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); pasted_text = data.get('pasted_text'); num_questions_req = data.get('num_questions', 5)
        if not pasted_text or not pasted_text.strip(): return jsonify({'status': 'error', 'message': 'Missing "pasted_text"'}), 400
        try: num_questions = int(num_questions_req); assert 2 <= num_questions <= 15
        except (ValueError, TypeError, AssertionError): return jsonify({'status': 'error', 'message': 'Invalid num_questions (must be int 2-15)'}), 400
        logging.info(f"Pasted Text Comp request: NumQuestions={num_questions}, Text Length={len(pasted_text)}")
        prompt_content = create_pasted_comprehension_prompt(pasted_text, num_questions)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=1500 + (num_questions * 100), temperature=0.7, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated Pasted Text Comp content length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'pasted_comprehension_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_pasted_comprehension: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_similar_questions', methods=['POST'])
def generate_similar_questions_route():
    logging.info("Received request at /generate_similar_questions")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); example_question = data.get('example_question'); num_questions_req = data.get('num_questions', 3); grade_level = data.get('grade_level', 'Not Specified')
        if not example_question or not example_question.strip(): return jsonify({'status': 'error', 'message': 'Missing "example_question"'}), 400
        try: num_questions = int(num_questions_req); assert 1 <= num_questions <= 10
        except (ValueError, TypeError, AssertionError): return jsonify({'status': 'error', 'message': 'Invalid num_questions (must be int 1-10)'}), 400
        logging.info(f"SimilarQ request: NumQuestions={num_questions}, Grade='{grade_level}', Example Length={len(example_question)}")
        prompt_content = create_similar_questions_prompt(example_question, num_questions, grade_level)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=1000 + (num_questions * 200), temperature=0.75, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated SimilarQ content length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'similar_questions_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_similar_questions: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_worksheet', methods=['POST']) # This is the Gap Fill route
def generate_worksheet_route():
    logging.info("Received request at /generate_worksheet (Gap Fill)")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); topic = data.get('topic'); grade_level = data.get('grade_level', 'middle school'); num_sentences_req = data.get('num_sentences', 7)
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic"'}), 400
        try: num_sentences = int(num_sentences_req); assert 3 <= num_sentences <= 15
        except (ValueError, TypeError, AssertionError): return jsonify({'status': 'error', 'message': 'Invalid num_sentences (must be int 3-15)'}), 400
        logging.info(f"Gap Fill request: Topic='{topic}', Grade='{grade_level}', NumSentences={num_sentences}")
        prompt_content = create_gap_fill_prompt(topic, grade_level, num_sentences)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=500 + (num_sentences * 60), temperature=0.7, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated Gap Fill content length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'worksheet_content': generated_content.strip()})
    except ValueError as ve: logging.warning(f"Validation Error: {ve}"); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_worksheet: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

# --- Word Search Routes ---
@app.route('/generate_word_list', methods=['POST'])
def generate_word_list_route():
    logging.info("Received request at /generate_word_list")
    if client is None: return jsonify({'status': 'error', 'message': 'Server error: AI client not initialized'}), 500
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); topic = data.get('topic'); grade_level = data.get('grade_level', 'middle school'); num_words = int(data.get('num_words', 15))
        if not topic: return jsonify({'status': 'error', 'message': 'Missing "topic"'}), 400
        if not 5 <= num_words <= 30: return jsonify({'status': 'error', 'message': 'Number of words must be between 5 and 30'}), 400
        logging.info(f"Word List request: Topic='{topic}', Grade='{grade_level}', NumWords={num_words}")
        prompt_content = create_word_list_prompt(topic, grade_level, num_words)
        message = client.messages.create(model=ANTHROPIC_MODEL_NAME, max_tokens=300 + (num_words * 10), temperature=0.5, messages=[{ "role": "user", "content": prompt_content }])
        if message.content and len(message.content) > 0 and hasattr(message.content[0], 'text'): generated_content = message.content[0].text.strip()
        else: raise ValueError("Failed to parse content from AI response.")
        logging.info(f"Generated word list raw text length: {len(generated_content)} chars")
        return jsonify({'status': 'success', 'word_list_text': generated_content })
    except ValueError as ve: logging.error(f"Validation/Processing Error: {ve}", exc_info=True); return jsonify({'status': 'error', 'message': str(ve)}), 400
    except (anthropic.APIError, httpx.RequestError) as e: return handle_anthropic_error(e)
    except Exception as e: logging.error(f"Unexpected error in /generate_word_list: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Internal server error.'}), 500

@app.route('/generate_word_search_grid', methods=['POST'])
def generate_word_search_grid_route():
    logging.info("Received request at /generate_word_search_grid")
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    try:
        data = request.get_json(); word_list_raw = data.get('word_list'); size_preference = data.get('size', 'medium'); allow_backwards = data.get('allow_backwards', False); mask = data.get('mask', None)
        if not word_list_raw or not isinstance(word_list_raw, list): return jsonify({'status': 'error', 'message': 'Missing or invalid "word_list"'}), 400
        size_map = {"small": 10, "medium": 13, "large": 15}; dimension = size_map.get(size_preference.lower(), 13)
        logging.info(f"Attempting grid generation: Size={dimension}x{dimension}, Backwards={allow_backwards}, Mask={mask}, Words={word_list_raw[:5]}...")
        grid_list, placed_words = generate_wordsearch_grid(wordlist=word_list_raw, nrows=dimension, ncols=dimension, allow_backwards_words=allow_backwards, mask_name=mask, attempts=15)
        html_output = '<div class="worksheet-section word-search-container">'; html_output += '<h3>Word Search</h3>'
        html_output += '<table class="word-search-grid" style="border-collapse: collapse; font-family: monospace; margin-bottom: 15px; border: 1px solid #ddd;">'
        for row_data in grid_list:
            html_output += '<tr>'
            for letter in row_data: display_letter = letter if letter.strip() else ' '; html_output += f'<td style="border: 1px solid #eee; width: 25px; height: 25px; text-align: center; vertical-align: middle; padding: 1px;">{display_letter}</td>'
            html_output += '</tr>'
        html_output += '</table>'; html_output += '<h3>Word List</h3>'
        html_output += '<ul class="word-search-list" style="list-style: none; padding-left: 0; columns: 2; -webkit-columns: 2; -moz-columns: 2;">'
        for word in sorted(placed_words): html_output += f'<li style="margin-bottom: 5px;">{word}</li>'
        html_output += '</ul></div>'; logging.info(f"Successfully generated word search grid HTML using internal logic.")
        return jsonify({'status': 'success', 'content_html': html_output})
    except (ValueError, PuzzleGenerationError) as gen_err: logging.warning(f"Word search generation failed: {gen_err}"); return jsonify({'status': 'error', 'message': str(gen_err)}), 400
    except Exception as e: logging.error(f"Unexpected error in /generate_word_search_grid: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'An internal server error occurred creating the word search.'}), 500

# --- Item/Worksheet Persistence Routes ---
@app.route('/save_item', methods=['POST'])
def save_item_route():
    logging.info("--- save_item route entered ---")
    if not request.is_json: logging.error("save_item: Request is not JSON"); return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    data = request.get_json(); logging.debug(f"save_item: Received data keys: {list(data.keys())}")
    item_type = data.get('item_type'); source_topic = data.get('source_topic'); source_url = data.get('source_url'); grade_level = data.get('grade_level'); content_html = data.get('content_html')
    if not item_type or not grade_level or not content_html: logging.error(f"save_item: Missing required fields. Got type: {item_type}, grade: {grade_level}, content: {bool(content_html)}"); return jsonify({'status': 'error', 'message': 'Missing required fields: item_type, grade_level, content_html'}), 400
    logging.debug(f"save_item: Preparing to create item: type={item_type}, grade={grade_level}")
    try:
        new_item = GeneratedItem(item_type=item_type, source_topic=source_topic, source_url=source_url, grade_level=grade_level, content_html=content_html)
        logging.debug(f"save_item: GeneratedItem object CREATED: {new_item}")
        db.session.add(new_item); logging.debug("save_item: Added to session.")
        db.session.commit(); logging.info(f"save_item: Commit successful. ID assigned: {new_item.id}")
        response_data = {'status': 'success', 'item_id': new_item.id}; logging.debug(f"save_item: Returning success data: {response_data}")
        return jsonify(response_data)
    except Exception as e: db.session.rollback(); logging.error(f"save_item: Exception during DB operation: {e}", exc_info=True); return jsonify({'status': 'error', 'message': f'Database error: {str(e)}'}), 500

@app.route('/get_item/<int:item_id>', methods=['GET'])
def get_item_route(item_id):
    logging.info(f"Received request at /get_item/{item_id}")
    try:
        item = db.session.get(GeneratedItem, item_id)
        if item is None: logging.warning(f"Item with ID {item_id} not found."); return jsonify({'status': 'error', 'message': 'Item not found in library.'}), 404
        logging.info(f"Found item: {item.item_type}")
        item_data = {'id': item.id, 'item_type': item.item_type, 'source_topic': item.source_topic, 'source_url': item.source_url,
                       'grade_level': item.grade_level, 'content_html': item.content_html, 'creation_date': item.creation_date.isoformat(),
                       'last_modified_date': item.last_modified_date.isoformat()}
        return jsonify({'status': 'success', 'item': item_data})
    except Exception as e: logging.error(f"Error retrieving item {item_id} from database: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Error retrieving item details.'}), 500

@app.route('/save_worksheet', methods=['POST'])
def save_worksheet_route():
    logging.info("Received request at /save_worksheet")
    if not request.is_json: return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    data = request.get_json(); worksheet_title = data.get('title', 'Untitled Worksheet'); item_ids_ordered = data.get('item_ids')
    if not item_ids_ordered or not isinstance(item_ids_ordered, list): return jsonify({'status': 'error', 'message': 'Missing or invalid "item_ids" list'}), 400
    logging.info(f"Attempting to save worksheet: Title='{worksheet_title}', Item IDs={item_ids_ordered}")
    try:
        new_worksheet = Worksheet(title=worksheet_title); db.session.add(new_worksheet); db.session.flush(); worksheet_id = new_worksheet.id
        items_to_add = []; order_index = 0
        for item_id in item_ids_ordered:
            generated_item = db.session.get(GeneratedItem, int(item_id))
            if generated_item:
                assoc_insert = worksheet_items_association.insert().values(worksheet_id=worksheet_id, generated_item_id=generated_item.id, item_order=order_index)
                db.session.execute(assoc_insert); items_to_add.append(generated_item.id); order_index += 1
            else: logging.warning(f"GeneratedItem ID {item_id} not found while saving worksheet {worksheet_id}. Skipping.")
        db.session.commit(); logging.info(f"Successfully saved Worksheet ID: {worksheet_id} with item IDs: {items_to_add} in order.")
        return jsonify({'status': 'success', 'message': 'Worksheet saved successfully!', 'worksheet_id': worksheet_id})
    except Exception as e: db.session.rollback(); logging.error(f"Error saving worksheet to database: {e}", exc_info=True); return jsonify({'status': 'error', 'message': f'Database error while saving worksheet: {e}'}), 500

@app.route('/list_worksheets', methods=['GET'])
def list_worksheets_route():
    logging.info("Received request at /list_worksheets")
    try:
        worksheets = Worksheet.query.order_by(Worksheet.last_modified_date.desc()).limit(50).all()
        worksheets_list = [{'id': ws.id, 'title': ws.title, 'creation_date': ws.creation_date.isoformat(),
                            'last_modified_date': ws.last_modified_date.isoformat()} for ws in worksheets]
        return jsonify({'status': 'success', 'worksheets': worksheets_list})
    except Exception as e: logging.error(f"Error retrieving worksheets from database: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Error retrieving worksheet list.'}), 500

@app.route('/load_worksheet/<int:worksheet_id>', methods=['GET'])
def load_worksheet_route(worksheet_id):
    logging.info(f"Received request at /load_worksheet/{worksheet_id}")
    try:
        worksheet = db.session.get(Worksheet, worksheet_id)
        if worksheet is None: return jsonify({'status': 'error', 'message': 'Worksheet not found.'}), 404
        ordered_items = worksheet.items # Relationship handles ordering
        items_data = [{'id': item.id, 'item_type': item.item_type, 'source_topic': item.source_topic, 'source_url': item.source_url,
                       'grade_level': item.grade_level, 'content_html': item.content_html} for item in ordered_items]
        logging.info(f"Returning {len(items_data)} items for Worksheet ID: {worksheet_id}")
        return jsonify({'status': 'success', 'worksheet_title': worksheet.title, 'items': items_data})
    except Exception as e: logging.error(f"Error retrieving worksheet {worksheet_id} from database: {e}", exc_info=True); return jsonify({'status': 'error', 'message': 'Error retrieving worksheet details.'}), 500

# --- Export and Frontend Routes ---
@app.route('/export/docx/<int:worksheet_id>')
def export_worksheet_docx(worksheet_id):
    """Exports a specific worksheet as a DOCX file."""
    logging.info(f"Request: /export/docx/{worksheet_id}") # DEBUG Line 1
    try:
        worksheet = Worksheet.query.get_or_404(worksheet_id)
        items = worksheet.items # Ordered based on relationship definition
        logging.info(f"Found worksheet '{worksheet.title}' with {len(items)} items.") # DEBUG Line 2

        document = docx.Document()
        logging.debug("Initialized docx Document.") # DEBUG Line 3

        # Apply Document Formatting
        target_font='Century Gothic'
        try:
            normal_style = document.styles['Normal']
            normal_style.font.name = target_font
            logging.info(f"Set Normal font to {target_font}") # DEBUG Line 4
        except Exception as font_e:
            logging.warning(f"Could not set default font to {target_font}: {font_e}")

        try:
            h3_style = document.styles['Heading 3']
            h3_style.font.name = target_font
            logging.info(f"Set H3 font to {target_font}") # DEBUG Line 5
        except Exception as style_e:
            logging.warning(f"Could not modify Heading 3 style: {style_e}")

        try:
            section = document.sections[0]
            section.left_margin=Inches(0.5); section.right_margin=Inches(0.5)
            section.top_margin=Inches(0.5); section.bottom_margin=Inches(0.5)
            logging.info("Set narrow margins.") # DEBUG Line 6
        except Exception as margin_e:
            logging.warning(f"Could not set margins: {margin_e}")

        # Add Worksheet Title
        document.add_heading(worksheet.title, level=1)
        document.add_paragraph() # Space after title
        logging.debug(f"Added worksheet title: {worksheet.title}") # DEBUG Line 7

        # --- Add Worksheet Items Loop ---
        logging.info("Starting item processing loop...") # DEBUG Line 8
        for item_index, item in enumerate(items):
            logging.debug(f"--- Processing item index: {item_index}, ID: {item.id}, Type: {item.item_type} ---") # DEBUG Line 9
            soup = BeautifulSoup(item.content_html, 'lxml')
            word_search_container = soup.find('div', class_='word-search-container')

            # Initialize variable for tracing
            processed_as_type = "Unknown" # DEBUG Line 10

            if word_search_container:
                # --- Word Search Specific Handling ---
                processed_as_type = "WordSearch" # DEBUG Line 11
                logging.debug("Item identified as Word Search.") # DEBUG Line 12

                # Add Heading (Word Search)
                heading_ws = word_search_container.find('h3')
                if heading_ws: document.add_heading(heading_ws.get_text(strip=True), level=3)
                else: document.add_heading("Word Search", level=3)
                logging.debug("Added WS Heading") # DEBUG Line 13

                # Generate and Add Grid Image
                grid_table = word_search_container.find('table', class_='word-search-grid')
                grid_list = []
                if grid_table:
                    for row in grid_table.find_all('tr'):
                        grid_list.append([cell.get_text(strip=True) for cell in row.find_all('td')])
                logging.debug(f"Extracted grid_list with {len(grid_list)} rows.") # DEBUG Line 14

                if grid_list:
                    grid_image = create_wordsearch_image(grid_list)
                    if grid_image:
                        img_buffer = io.BytesIO(); grid_image.save(img_buffer, format='PNG'); img_buffer.seek(0)
                        try:
                            document.add_picture(img_buffer, width=Inches(6.0))
                            logging.debug("Added word search grid image.") # DEBUG Line 15
                        except Exception as pic_e:
                            logging.error(f"Failed to add picture: {pic_e}")
                            document.add_paragraph("[Error adding grid image]", style='Comment')
                    else:
                        logging.warning("create_wordsearch_image returned None.") # DEBUG Line 16
                        document.add_paragraph("[Error generating grid image]", style='Comment')
                else:
                     logging.warning("Word search grid table not found in HTML.") # DEBUG Line 17
                     document.add_paragraph("[Word search grid table not found in HTML]", style='Comment')

                # Add Word List section
                all_headings = word_search_container.find_all('h3') # Find all h3s
                if len(all_headings) > 1: # Check if second h3 exists (Word List heading)
                    document.add_heading(all_headings[1].get_text(strip=True), level=3)
                else:
                     document.add_heading("Word List", level=3) # Fallback heading

                word_list_ul = word_search_container.find('ul', class_='word-search-list')
                words_to_find = [] # Create an empty list to store words
                if word_list_ul:
                    for li in word_list_ul.find_all('li'):
                        word = li.get_text(strip=True)
                        if word: # Make sure word is not empty
                            words_to_find.append(word) # Add word to the list
                    logging.debug(f"Extracted {len(words_to_find)} words for list.")
                else:
                    logging.warning("Word search word list UL element not found in HTML for export.")

                # Check if we actually found any words
                if words_to_find:
                    # Join the list into a single string with "     " separator
                    joined_words_string = "     ".join(words_to_find)
                    # Add the joined string as a single paragraph
                    document.add_paragraph(joined_words_string)
                    logging.debug(f"Added joined word list paragraph: {joined_words_string[:100]}...")
                else:
                    # If the UL was missing or empty, add a placeholder comment
                    document.add_paragraph("[Word list empty or not found in HTML]", style='Comment')

                document.add_paragraph() # Space after word search item finished

            else:
                # --- Original/Generic Handling for OTHER item types ---
                processed_as_type = "Generic" # DEBUG Line 21
                logging.debug("Item identified as Generic type.") # DEBUG Line 22
                content_elements = list(soup.find('body').find_all(True, recursive=False)) if soup.find('body') else list(soup.find_all(True, recursive=False))
                logging.debug(f"Assigned content_elements. Length: {len(content_elements)}") # DEBUG Line 23

                if not content_elements:
                    plain_text = soup.get_text(strip=True)
                    if plain_text:
                         logging.debug("No elements found, adding plain text only.") # DEBUG Line 24
                         document.add_paragraph(plain_text)
                    else:
                         logging.warning(f"Item ID {item.id} (Generic) has no content elements or text.") # DEBUG Line 25
                else:
                    logging.debug(f"Entering loop for {len(content_elements)} generic elements...") # DEBUG Line 26
                    # Loop through content_elements HERE, INSIDE the else block
                    for element_index, element in enumerate(content_elements): # DEBUG Added index
                        logging.debug(f"  Processing generic element index {element_index}: Name={element.name}, Class={element.get('class',[])}") # DEBUG Line 27
                        if element.name == 'p':
                            para=document.add_paragraph(); add_runs_from_html_element(para, element)
                        elif element.name in ['ol', 'ul']:
                            style = 'List Number' if element.name == 'ol' else 'List Bullet'
                            list_items_found = element.find_all('li', recursive=False) # DEBUG
                            logging.debug(f"    Found list '{element.name}' with {len(list_items_found)} items.") # DEBUG Line 28
                            for li in list_items_found:
                                 para=document.add_paragraph(style=style); add_runs_from_html_element(para, li)
                            # Attempt numbering reset with tiny font paragraph
                            p=document.add_paragraph(); r=p.add_run(); r.font.size = Pt(1)
                            logging.debug("    Added tiny-font paragraph break after list.") # DEBUG Line 29
                        elif element.name == 'h3':
                             document.add_heading(element.get_text(strip=True), level=3)
                        elif element.name == 'div' and 'worksheet-section' in element.get('class', []):
                             logging.debug("    Processing div.worksheet-section") # DEBUG Line 30
                             if h:=element.find('h3'): document.add_heading(h.get_text(strip=True), level=3)
                             for inner in element.find_all(['p','ol','ul'], recursive=False):
                                  logging.debug(f"      Processing inner element: {inner.name}") # DEBUG Line 31
                                  if inner.name == 'p':
                                       para=document.add_paragraph(); add_runs_from_html_element(para, inner)
                                  elif inner.name in ['ol', 'ul']:
                                       p=document.add_paragraph(); r=p.add_run(); r.font.size = Pt(1) # Reset attempt
                                       style = 'List Number' if inner.name == 'ol' else 'List Bullet'
                                       inner_list_items_found = inner.find_all('li', recursive=False) # DEBUG
                                       logging.debug(f"        Found inner list '{inner.name}' with {len(inner_list_items_found)} items.") # DEBUG Line 32
                                       for li in inner_list_items_found:
                                            para=document.add_paragraph(style=style); add_runs_from_html_element(para, li)
                        elif element.name is None and element.string and element.string.strip():
                             logging.debug(f"    Found text node: {element.string.strip()[:30]}...") # DEBUG Line 33
                             document.add_paragraph(element.string.strip())
                        else:
                             logging.debug(f"    Ignoring generic element: Name={element.name}, Class={element.get('class',[])}") # DEBUG Line 34
                    logging.debug("  Finished loop for generic elements.") # DEBUG Line 35
            # --- End of the 'else' block for generic handling ---

            logging.debug(f"End processing item index {item_index}. Identified as: {processed_as_type}") # DEBUG Line 36
            # --- Add separator BETWEEN items ---
            if item_index < len(items) - 1:
                logging.debug(f"Adding separator after item index {item_index}") # DEBUG Line 37
                document.add_paragraph("_________________________")
                document.add_paragraph()
        # --- End of main item loop ---
        logging.info("Finished item processing loop.") # DEBUG Line 38

        # --- Save and Return File ---
        logging.debug("Saving document to buffer...") # DEBUG Line 39
        file_stream = io.BytesIO(); document.save(file_stream); file_stream.seek(0)
        filename = f"{worksheet.title.replace(' ','_').lower() or 'worksheet'}.docx"
        logging.info(f"Sending DOCX file: {filename}") # DEBUG Line 40
        return send_file(file_stream, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')

    except Exception as e:
        # Log the error properly in a real app
        logging.error(f"Error exporting DOCX for worksheet {worksheet_id}", exc_info=True) # DEBUG Line 41 (includes traceback)
        # Return a user-friendly error page or message
        return f"Error exporting worksheet: {str(e)}", 500

# --- Helper function add_runs_from_html_element remains the same ---
# def add_runs_from_html_element(paragraph, element):
#    ... (keep the previous version) ...500

def add_runs_from_html_element(paragraph, element):
    """Adds formatted runs to a python-docx paragraph based on basic HTML tags."""
    for content in element.contents:
        if isinstance(content, str): paragraph.add_run(content.replace('\xa0', ' '))
        elif content.name in ['strong', 'b']: r=paragraph.add_run(content.get_text(strip=True).replace('\xa0', ' ')); r.bold = True
        elif content.name in ['em', 'i']: r=paragraph.add_run(content.get_text(strip=True).replace('\xa0', ' ')); r.italic = True
        elif content.name == 'span' and 'gap-placeholder' in content.get('class', []): paragraph.add_run(" [__________] ")
        elif content.name == 'br': paragraph.add_run("\n")

@app.route('/')
def serve_index():
    """Serves the index.html file."""
    logging.info("Serving index.html")
    return send_from_directory('.', 'index.html')

# --- Run the App ---
if __name__ == '__main__':
    # Note: debug=True is useful locally but should be False in production
    app.run(host='127.0.0.1', port=5001, debug=True)