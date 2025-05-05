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
import json
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

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
    db.metadata,
    db.Column('worksheet_id', db.Integer, db.ForeignKey('worksheet.id'), primary_key=True),
    db.Column('generated_item_id', db.Integer, db.ForeignKey('generated_item.id'), primary_key=True), # Matches your column name
    db.Column('item_order', db.Integer, nullable=False)
)

# GeneratedItem Model
class GeneratedItem(db.Model):
    __tablename__ = 'generated_item'
    id = db.Column(db.Integer, primary_key=True)
    item_type = db.Column(db.String(50), nullable=False)
    source_topic = db.Column(db.String(250), nullable=True)
    source_url = db.Column(db.String(500), nullable=True)
    grade_level = db.Column(db.String(50), nullable=False)
    content_html = db.Column(db.Text, nullable=False)
    item_data_json = db.Column(db.Text, nullable=True)
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
    __tablename__ = 'worksheet'
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

@app.route('/generate_keywords_definitions', methods=['POST'])
def generate_keywords_definitions_route():
    logging.info("Request received: /generate_keywords_definitions")
    if not request.is_json:
        logging.warning("Request aborted: Content-Type is not application/json.")
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    data = request.get_json()
    if not data:
        logging.warning("Request aborted: Empty JSON payload received.")
        return jsonify({'status': 'error', 'message': 'Empty JSON payload received.'}), 400

    topic = data.get('topic')
    grade_level = data.get('grade_level')
    try:
        # Default to 10 keywords if not provided or invalid
        num_keywords = int(data.get('num_keywords', 10))
        if num_keywords <= 0:
            num_keywords = 10
            logging.warning("Invalid num_keywords <= 0 received, defaulting to 10.")
    except (ValueError, TypeError):
        num_keywords = 10
        logging.warning("Invalid or missing num_keywords received, defaulting to 10.")


    if not topic or not grade_level:
        logging.warning("Request aborted: Missing 'topic' or 'grade_level' in JSON payload.")
        return jsonify({'status': 'error', 'message': 'Missing "topic" or "grade_level".'}), 400

    logging.debug(f"Parameters received: topic='{topic}', grade_level='{grade_level}', num_keywords={num_keywords}")

    # --- Construct the AI Prompt ---
    prompt = f"""Generate {num_keywords} unique keywords related to the topic "{topic}" suitable for grade level {grade_level}.
For each keyword, provide a concise and clear definition appropriate for that grade level.
Format the output STRICTLY as a JSON list of objects, where each object has a "keyword" key and a "definition" key.
Do not include any other text, explanations, or markdown formatting before or after the JSON list. Just the raw JSON.

Example format:
[
  {{"keyword": "Example Term 1", "definition": "This is the definition for term 1."}},
  {{"keyword": "Example Term 2", "definition": "This is the definition for term 2."}}
]

Begin JSON list now:"""

    try:
        logging.info(f"Sending request to Anthropic API for keywords/definitions...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME, # Use your configured model name variable
            max_tokens=1500,  # Adjust as needed, maybe calculate based on num_keywords?
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract the text content from the response
        # Assuming the response structure gives text in message.content[0].text
        if not message.content or not message.content[0].text:
             raise ValueError("Received empty content from Anthropic API.")

        ai_response_text = message.content[0].text.strip()
        logging.debug(f"Raw AI response received:\n{ai_response_text}")

        # --- Parse the JSON Response ---
        # Sometimes the AI might still wrap the JSON in markdown backticks
        if ai_response_text.startswith("```json"):
            ai_response_text = ai_response_text[7:] # Remove ```json
        if ai_response_text.endswith("```"):
            ai_response_text = ai_response_text[:-3] # Remove ```
        ai_response_text = ai_response_text.strip() # Clean whitespace again

        try:
            keyword_data = json.loads(ai_response_text)
        except json.JSONDecodeError as json_err:
            logging.error(f"Failed to parse JSON from AI response: {json_err}")
            logging.error(f"Problematic AI response text: {ai_response_text}")
            # Consider trying to extract JSON manually if needed, but often better to fail
            return jsonify({'status': 'error', 'message': f'AI response format error (not valid JSON).'}), 500

        # --- Basic Validation of Parsed Data ---
        if not isinstance(keyword_data, list):
            logging.error(f"AI response parsed, but it's not a list. Type: {type(keyword_data)}")
            return jsonify({'status': 'error', 'message': 'AI response format error (expected a list).'}), 500

        validated_data = []
        for item in keyword_data:
            if isinstance(item, dict) and 'keyword' in item and 'definition' in item:
                # Basic cleaning
                kw = str(item['keyword']).strip()
                df = str(item['definition']).strip()
                if kw and df: # Ensure they are not empty after stripping
                     validated_data.append({'keyword': kw, 'definition': df})
            else:
                logging.warning(f"Skipping invalid item in AI response list: {item}")

        if not validated_data:
             logging.error("AI response parsed as list, but no valid keyword/definition pairs found.")
             return jsonify({'status': 'error', 'message': 'AI returned no valid keyword/definition pairs.'}), 500

        logging.info(f"Successfully generated and parsed {len(validated_data)} keywords/definitions.")
        return jsonify({'status': 'success', 'keywords': validated_data})

    except Exception as e:
        # Catch errors from the API call itself or other unexpected issues
        logging.error(f"Error during keywords/definitions generation: {e}", exc_info=True)
        # Provide a generic error to the frontend
        return jsonify({'status': 'error', 'message': 'An internal error occurred while generating keywords.'}), 500

# --- Item/Worksheet Persistence Routes ---
@app.route('/save_item', methods=['POST'])
def save_item_route():
    logging.info("--- save_item route entered ---")
    if not request.is_json: logging.error("save_item: Request is not JSON"); return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
    data = request.get_json(); logging.debug(f"save_item: Received data keys: {list(data.keys())}")
    item_type = data.get('item_type'); source_topic = data.get('source_topic'); source_url = data.get('source_url'); grade_level = data.get('grade_level'); item_data_json = data.get('item_data_json', None); content_html = data.get('content_html')
    if not item_type or not grade_level or not content_html: logging.error(f"save_item: Missing required fields. Got type: {item_type}, grade: {grade_level}, content: {bool(content_html)}"); return jsonify({'status': 'error', 'message': 'Missing required fields: item_type, grade_level, content_html'}), 400
    logging.debug(f"save_item: Preparing to create item: type={item_type}, grade={grade_level}")
    try:
        new_item = GeneratedItem(item_type=item_type, source_topic=source_topic, source_url=source_url, grade_level=grade_level, item_data_json=item_data_json, content_html=content_html)
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
                       'grade_level': item.grade_level, 'item_data_json': item.item_data_json, 'content_html': item.content_html, 'creation_date': item.creation_date.isoformat(),
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
@app.route('/generate_similar_written_questions', methods=['POST'])
def generate_similar_written_questions_route():
    logging.info("Request received: /generate_similar_written_questions")
    if not request.is_json:
        logging.warning("Request aborted: Content-Type is not application/json.")
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

    data = request.get_json()
    if not data:
        logging.warning("Request aborted: Empty JSON payload received.")
        return jsonify({'status': 'error', 'message': 'Empty JSON payload received.'}), 400

    example_question = data.get('example_question')
    grade_level = data.get('grade_level') # Might be "Not Specified"
    try:
        num_questions = int(data.get('num_questions', 3))
        if num_questions <= 0:
            num_questions = 3
            logging.warning("Invalid num_questions <= 0 received, defaulting to 3.")
    except (ValueError, TypeError):
        num_questions = 3
        logging.warning("Invalid or missing num_questions received, defaulting to 3.")

    if not example_question:
        logging.warning("Request aborted: Missing 'example_question' in JSON payload.")
        return jsonify({'status': 'error', 'message': 'Missing "example_question".'}), 400

    # Use grade level if provided, otherwise let AI infer
    grade_level_text = f"potentially intended for grade level '{grade_level}'" if grade_level and grade_level != "Not Specified" else "with an unspecified grade level"

    logging.debug(f"Parameters: num_questions={num_questions}, grade_level='{grade_level}', example='{example_question[:100]}...'")

    # --- Construct the AI Prompt ---
    prompt = f"""You are an expert assessment designer tasked with creating question variations.
Analyze the following example question, {grade_level_text}:

--- Example Question ---
{example_question}
--- End Example Question ---

First, identify and concisely describe the primary skill(s) being assessed by this question.
Second, generate {num_questions} new, distinct questions that assess the *exact same skill(s)* you identified. Ensure these new questions are appropriate for the specified grade level (or maintain the implied level of the example), keep a similar difficulty, but use different scenarios, contexts, or specific details. Avoid simple rephrasing of the example.

Format your response strictly as a JSON object containing two keys:
1. "skills_identified": A string describing the core skill(s) tested (keep this brief, maybe 1-2 sentences).
2. "similar_questions": A JSON list containing exactly {num_questions} strings, where each string is one of the new questions.

Do not include any explanations or text outside the JSON object.

Example JSON Output Format:
{{
  "skills_identified": "Applying the definition of photosynthesis to identify necessary components.",
  "similar_questions": [
    "Besides sunlight, what is one essential reactant that plants need for photosynthesis?",
    "Which part of a plant cell is primarily responsible for carrying out photosynthesis?",
    "If a plant is kept in complete darkness but given water and carbon dioxide, can it perform photosynthesis? Explain why or why not."
  ]
}}

Begin JSON object now:"""

    try:
        logging.info(f"Sending request to Anthropic API for similar written questions...")
        message = client.messages.create(
            model=ANTHROPIC_MODEL_NAME,
            max_tokens=1500, # Adjust as needed
            messages=[{"role": "user", "content": prompt}]
        )

        if not message.content or not message.content[0].text:
             raise ValueError("Received empty content from Anthropic API.")

        ai_response_text = message.content[0].text.strip()
        logging.debug(f"Raw AI response received:\n{ai_response_text}")

        # Clean potential markdown backticks
        if ai_response_text.startswith("```json"):
            ai_response_text = ai_response_text[7:]
        if ai_response_text.endswith("```"):
            ai_response_text = ai_response_text[:-3]
        ai_response_text = ai_response_text.strip()

        try:
            parsed_data = json.loads(ai_response_text)
        except json.JSONDecodeError as json_err:
            logging.error(f"Failed to parse JSON from AI response: {json_err}")
            logging.error(f"Problematic AI response text for similar written q: {ai_response_text}")
            return jsonify({'status': 'error', 'message': 'AI response format error (not valid JSON).'}), 500

        # Validate structure
        if not isinstance(parsed_data, dict):
             logging.error(f"Parsed data is not a dict: {type(parsed_data)}")
             return jsonify({'status': 'error', 'message': 'AI response format error (expected JSON object).'}), 500

        skills = parsed_data.get("skills_identified")
        questions = parsed_data.get("similar_questions")

        if not isinstance(skills, str) or not isinstance(questions, list):
             logging.error(f"Missing/invalid keys in parsed JSON. Skills type: {type(skills)}, Questions type: {type(questions)}")
             return jsonify({'status': 'error', 'message': 'AI response format error (missing/invalid keys).'}), 500

        # Optional: Validate number of questions returned matches request?
        # if len(questions) != num_questions:
        #    logging.warning(f"AI returned {len(questions)} questions, expected {num_questions}")

        validated_questions = [str(q).strip() for q in questions if isinstance(q, str) and str(q).strip()]

        logging.info(f"Successfully generated {len(validated_questions)} similar written questions.")
        # Return the whole parsed object nested under 'data'
        return jsonify({
            'status': 'success',
            'data': {
                 'skills_identified': skills.strip(),
                 'similar_questions': validated_questions
                 }
            })

    except Exception as e:
        logging.error(f"Error during similar written questions generation: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'An internal error occurred while generating questions.'}), 500
# --- Export and Frontend Routes ---
@app.route('/export/docx/<int:worksheet_id>')
def export_worksheet_docx(worksheet_id):
    """Exports a specific worksheet as a DOCX file."""
    logging.info(f"Request: /export/docx/{worksheet_id}")
    try:
        # Use eager loading to get items efficiently if possible, or default relationship loading
        # worksheet = Worksheet.query.options(selectinload(Worksheet.items)).get_or_404(worksheet_id) # More advanced
        worksheet = Worksheet.query.get_or_404(worksheet_id)
        # Access items (SQLAlchemy should handle the ordering defined in the relationship)
        items = worksheet.items
        logging.info(f"Found worksheet '{worksheet.title}' with {len(items)} items.")

        document = docx.Document()
        logging.debug("Initialized docx Document.")

        # --- Apply Document Formatting ---
        target_font='Century Gothic'
        try:
            normal_style = document.styles['Normal']
            normal_style.font.name = target_font
            normal_style.paragraph_format.space_after = Pt(6) # Add some space after paragraphs
            logging.info(f"Set Normal font to {target_font} and paragraph spacing.")
        except Exception as font_e:
            logging.warning(f"Could not set default font/style: {font_e}")

        try:
            h3_style = document.styles['Heading 3']
            h3_style.font.name = target_font
            h3_style.paragraph_format.space_before = Pt(12)
            h3_style.paragraph_format.space_after = Pt(6)
            logging.info(f"Set H3 font to {target_font} and spacing.")
        except Exception as style_e:
            logging.warning(f"Could not modify Heading 3 style: {style_e}")

        try: # Set Margins
            section = document.sections[0]
            section.left_margin=Inches(0.5); section.right_margin=Inches(0.5)
            section.top_margin=Inches(0.5); section.bottom_margin=Inches(0.5)
            logging.info("Set narrow margins.")
        except Exception as margin_e:
            logging.warning(f"Could not set margins: {margin_e}")

        # --- Add Worksheet Title ---
        document.add_heading(worksheet.title, level=1)
        document.add_paragraph() # Space after title
        logging.debug(f"Added worksheet title: {worksheet.title}")

        # --- Add Worksheet Items Loop ---
        logging.info("Starting item processing loop...")
        for item_index, item in enumerate(items):
            logging.debug(f"--- Processing item index: {item_index}, ID: {item.id}, Type: {item.item_type} ---")
            soup = BeautifulSoup(item.content_html, 'lxml') # Use lxml for better parsing
            processed_as_type = "Unknown" # Initialize tracker

            # Check for specific structures first
            word_search_container = soup.find('div', class_='word-search-container')

            # --- Condition 1: Word Search ---
            if word_search_container:
                processed_as_type = "WordSearch"
                logging.debug("Item identified as Word Search.")
                try:
                    # Add Heading (Word Search)
                    heading_ws = word_search_container.find('h3') # Find first h3
                    document.add_heading(heading_ws.get_text(strip=True) if heading_ws else "Word Search", level=3)
                    logging.debug("Added WS Heading")

                    # Extract grid data
                    grid_table = word_search_container.find('table', class_='word-search-grid')
                    grid_list = []
                    if grid_table:
                        for row in grid_table.find_all('tr'):
                            grid_list.append([cell.get_text(strip=True) for cell in row.find_all('td')])
                    logging.debug(f"Extracted grid_list with {len(grid_list)} rows.")

                    # Add Grid Image
                    if grid_list:
                        grid_image = create_wordsearch_image(grid_list) # Assumes this function exists
                        if grid_image:
                            img_buffer = io.BytesIO(); grid_image.save(img_buffer, format='PNG'); img_buffer.seek(0)
                            try: document.add_picture(img_buffer, width=Inches(6.0))
                            except Exception as pic_e: logging.error(f"Failed to add picture: {pic_e}"); document.add_paragraph("[Error adding grid image]", style='Comment')
                            logging.debug("Added word search grid image (or error placeholder).")
                        else: logging.warning("create_wordsearch_image returned None."); document.add_paragraph("[Error generating grid image]", style='Comment')
                    else: logging.warning("Word search grid table not found in HTML."); document.add_paragraph("[Word search grid table not found in HTML]", style='Comment')

                    # Add Word List Heading
                    all_headings = word_search_container.find_all('h3')
                    document.add_heading(all_headings[1].get_text(strip=True) if len(all_headings) > 1 else "Word List", level=3)

                    # Add Word List Paragraph
                    word_list_ul = word_search_container.find('ul', class_='word-search-list')
                    words_to_find = [li.get_text(strip=True) for li in word_list_ul.find_all('li')] if word_list_ul else []
                    words_to_find = [word for word in words_to_find if word] # Filter empty strings
                    logging.debug(f"Extracted {len(words_to_find)} words for list.")

                    if words_to_find:
                        joined_words_string = "     ".join(words_to_find)
                        document.add_paragraph(joined_words_string)
                        logging.debug(f"Added joined word list paragraph.")
                    else: document.add_paragraph("[Word list empty or not found in HTML]", style='Comment')

                    document.add_paragraph() # Space after word search item
                except Exception as ws_err:
                    logging.error(f"Error processing WordSearch item {item.id}: {ws_err}", exc_info=True)
                    document.add_paragraph(f"[Error processing Word Search item {item.id}]", style='Comment')

            # --- Condition 2: Keywords Table ---
            elif item.item_type.startswith('keywords-'):
                processed_as_type = "KeywordsTable"
                logging.debug(f"Processing keywords item {item.id} (Type: {item.item_type}) for DOCX.")
                try:
                    soup_kw = BeautifulSoup(item.content_html, 'lxml') # Re-parse just in case soup object was modified
                    html_table = soup_kw.find('table', class_='keywords-table')

                    if not html_table: raise ValueError("Could not find 'table.keywords-table' in HTML")

                    rows = html_table.find_all('tr')
                    if not rows: raise ValueError("Keywords table found but contained no rows.")

                    header_cells = rows[0].find_all(['th', 'td'])
                    num_cols = len(header_cells)
                    if num_cols <= 0: raise ValueError("Keywords table found, but header row had no columns.")

                    data_rows = rows[1:]
                    num_data_rows = len(data_rows)

                    logging.debug(f"Creating DOCX table with {num_data_rows + 1} rows and {num_cols} cols.")
                    docx_table = document.add_table(rows=num_data_rows + 1, cols=num_cols)
                    docx_table.style = 'Table Grid'

                    # Populate Header
                    for j, cell in enumerate(header_cells):
                         if j < num_cols:
                            header_text = ' '.join(cell.get_text(strip=True).split())
                            hdr_cell = docx_table.cell(0, j)
                            hdr_cell.text = header_text
                            if hdr_cell.paragraphs and hdr_cell.paragraphs[0].runs: hdr_cell.paragraphs[0].runs[0].font.bold = True
                            elif hdr_cell.paragraphs: hdr_cell.paragraphs[0].add_run().font.bold = True

                    # Populate Data
                    for i, html_row in enumerate(data_rows):
                        cells = html_row.find_all('td')
                        for j, cell in enumerate(cells):
                             if j < num_cols:
                                cell_text = ' '.join(cell.get_text(strip=True).split())
                                docx_table.cell(i + 1, j).text = cell_text if cell_text else ""

                    # Modify Borders for Matching Type
                    if item.item_type == 'keywords-matching' and num_cols == 3:
                        logging.debug(f"Modifying middle column borders for item {item.id}.")
                        middle_col_index = 1
                        total_rows = num_data_rows + 1
                        for i in range(total_rows):
                            try:
                                cell = docx_table.cell(i, middle_col_index)
                                tcPr = cell._tc.get_or_add_tcPr()
                                tcBorders = tcPr.first_child_found_in("w:tcBorders")
                                if tcBorders is None: tcBorders = OxmlElement("w:tcBorders"); tcPr.append(tcBorders)
                                top_border = OxmlElement('w:top'); top_border.set(qn('w:val'), 'nil'); tcBorders.append(top_border)
                                bottom_border = OxmlElement('w:bottom'); bottom_border.set(qn('w:val'), 'nil'); tcBorders.append(bottom_border)
                            except IndexError: logging.warning(f"IndexError accessing cell ({i},{middle_col_index}) modifying borders.")
                            except Exception as border_err: logging.error(f"Error modifying borders cell({i},{middle_col_index}): {border_err}")
                        logging.debug(f"Finished border removal attempt for item {item.id}.")
                    elif item.item_type == 'keywords-matching' and num_cols != 3:
                         logging.warning(f"Keyword matching item {item.id} did not have 3 columns. Borders not modified.")

                    document.add_paragraph() # Space after table
                    logging.debug(f"Successfully added keywords table for item {item.id} to DOCX.")

                except ValueError as ve: # Catch specific parsing value errors
                     logging.warning(f"ValueError processing keywords table for item {item.id}: {ve}")
                     document.add_paragraph(f"[Keyword table content invalid for item {item.id}: {ve}]", style='Comment')
                except Exception as parse_err: # Catch other unexpected errors
                    logging.error(f"Error processing keywords table item {item.id}: {parse_err}", exc_info=True)
                    document.add_paragraph(f"[Error processing keyword table for item {item.id}]", style='Comment')

            # --- Condition 3: Similar Written Questions ---
            elif item.item_type == 'similarWrittenQ':
                processed_as_type = "SimilarWrittenQ"
                logging.debug(f"Processing similarWrittenQ item {item.id} for DOCX.")
                skills = "Skills not specified."
                questions = []
                try:
                    if item.item_data_json:
                        parsed_data = json.loads(item.item_data_json)
                        if isinstance(parsed_data, dict):
                            skills = parsed_data.get('skills_identified', skills).strip()
                            questions_raw = parsed_data.get('similar_questions', [])
                            if isinstance(questions_raw, list):
                                questions = [str(q).strip() for q in questions_raw if str(q).strip()]
                            else: logging.warning(f"similar_questions in JSON for item {item.id} was not a list.")
                        else: logging.warning(f"Parsed item_data_json for item {item.id} was not a dict.")
                    else: logging.warning(f"item_data_json not found for similarWrittenQ item {item.id}.")

                    # Add content to DOCX
                    document.add_heading("Skills Tested", level=3)
                    document.add_paragraph(skills)
                    document.add_paragraph()

                    document.add_heading("Generated Questions", level=3)
                    if questions:
                        for q_text in questions:
                            document.add_paragraph(q_text, style='List Number')
                        p=document.add_paragraph(); r=p.add_run(); r.font.size = Pt(1) # Reset numbering
                    else: document.add_paragraph("[No similar questions found in saved data]", style='Comment')

                    document.add_paragraph() # Spacing after section
                    logging.debug(f"Successfully added SimilarWrittenQ item {item.id} to DOCX.")

                except json.JSONDecodeError as json_err:
                     logging.error(f"Failed to parse item_data_json for similarWrittenQ item {item.id}: {json_err}")
                     document.add_paragraph(f"[Error reading saved data for Similar Written Questions item {item.id}]", style='Comment')
                except Exception as export_err:
                    logging.error(f"Error adding similarWrittenQ item {item.id} to DOCX: {export_err}", exc_info=True)
                    document.add_paragraph(f"[Error processing Similar Written Questions item {item.id}]", style='Comment')

            # --- Condition 4: Generic/Fallback Handling ---
            else:
                processed_as_type = "Generic"
                logging.debug("Item identified as Generic type. Parsing content_html.")
                try:
                    # Find top-level elements, excluding head/body/html if present
                    body = soup.find('body')
                    if body: content_elements = body.find_all(True, recursive=False)
                    else: content_elements = soup.find_all(True, recursive=False)

                    # If still no elements, try getting text
                    if not content_elements:
                        plain_text = soup.get_text(strip=True)
                        if plain_text:
                             logging.debug("No elements found, adding plain text only.")
                             document.add_paragraph(plain_text)
                        else: logging.warning(f"Item ID {item.id} (Generic) has no content elements or text.")
                    else:
                        logging.debug(f"Entering loop for {len(content_elements)} generic elements...")
                        for element in content_elements:
                            logging.debug(f"  Processing generic element: Name={element.name}, Class={element.get('class',[])}")
                            if element.name == 'p':
                                para=document.add_paragraph(); add_runs_from_html_element(para, element)
                            elif element.name in ['ol', 'ul']:
                                style = 'List Number' if element.name == 'ol' else 'List Bullet'
                                list_items = element.find_all('li', recursive=False)
                                logging.debug(f"    Found list '{element.name}' with {len(list_items)} items.")
                                for li in list_items:
                                     para=document.add_paragraph(style=style); add_runs_from_html_element(para, li)
                                # Attempt numbering reset
                                p=document.add_paragraph(); r=p.add_run(); r.font.size = Pt(1)
                                logging.debug("    Added tiny-font paragraph break after list.")
                            elif element.name == 'h3':
                                 document.add_heading(element.get_text(strip=True), level=3)
                            elif element.name == 'div' and 'worksheet-section' in element.get('class', []):
                                 logging.debug("    Processing div.worksheet-section")
                                 if h:=element.find('h3'): document.add_heading(h.get_text(strip=True), level=3)
                                 # Process inner elements within the section div
                                 for inner in element.find_all(['p','ol','ul'], recursive=False):
                                      logging.debug(f"      Processing inner element: {inner.name}")
                                      if inner.name == 'p': para=document.add_paragraph(); add_runs_from_html_element(para, inner)
                                      elif inner.name in ['ol', 'ul']:
                                           p=document.add_paragraph(); r=p.add_run(); r.font.size = Pt(1) # Reset attempt
                                           style = 'List Number' if inner.name == 'ol' else 'List Bullet'
                                           inner_list_items = inner.find_all('li', recursive=False)
                                           logging.debug(f"        Found inner list '{inner.name}' with {len(inner_list_items)} items.")
                                           for li in inner_list_items: para=document.add_paragraph(style=style); add_runs_from_html_element(para, li)
                            # Handle potential top-level text nodes if needed (less common)
                            # elif element.name is None and element.string and element.string.strip():
                            #     document.add_paragraph(element.string.strip())
                            else:
                                 logging.debug(f"    Ignoring generic element: Name={element.name}")
                        logging.debug("  Finished loop for generic elements.")
                except Exception as generic_err:
                     logging.error(f"Error processing generic item {item.id}: {generic_err}", exc_info=True)
                     document.add_paragraph(f"[Error processing content for item {item.id}]", style='Comment')

            # --- End of the main if/elif/else block for item types ---
            logging.debug(f"End processing item index {item_index}. Identified as: {processed_as_type}")

            # --- Add separator BETWEEN items ---
            if item_index < len(items) - 1:
                logging.debug(f"Adding separator after item index {item_index}")
                # Use a more subtle separator? Like a paragraph break or horizontal rule if supported well
                document.add_paragraph().add_run().add_break(docx.enum.text.WD_BREAK.PAGE) # Or just PAGE break
                # document.add_paragraph("---") # Simple separator
                document.add_paragraph()

        # --- End of main item loop ---
        logging.info("Finished item processing loop.")

        # --- Save and Return File ---
        logging.debug("Saving document to buffer...")
        file_stream = io.BytesIO()
        document.save(file_stream)
        file_stream.seek(0)
        # Sanitize title for filename
        safe_title = re.sub(r'[\\/*?:"<>|]', "", worksheet.title) # Remove invalid chars
        safe_title = re.sub(r'\s+', '_', safe_title) # Replace whitespace with underscore
        filename = f"{safe_title.lower() or 'worksheet'}.docx"
        logging.info(f"Sending DOCX file: {filename}")
        return send_file(
            file_stream,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        logging.error(f"CRITICAL Error exporting DOCX for worksheet {worksheet_id}", exc_info=True)
        # Return a user-friendly error page or message in production
        # For debugging, returning the error might be useful
        return f"Error exporting worksheet: {str(e)}<br><pre>{traceback.format_exc()}</pre>", 500

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