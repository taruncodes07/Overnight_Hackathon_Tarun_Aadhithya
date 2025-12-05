import re
import unicodedata
import pandas as pd
import difflib

# --- Global Configuration and Data Loading ---

SIMILARITY_THRESHOLD = 0.80 

# Load and process the CSV data
df = pd.read_csv("hindi top 150 swear words .csv", header=None)
OFFENSIVE_TERMS = df[0].astype(str).str.lower().str.strip().tolist()

# Add manually verified variants for robust detection of obfuscation
OFFENSIVE_TERMS.extend(["kute ki aulad", "lavde", "lode", "madarchod", "chutiya"]) 

# Use a Set to ensure only unique terms are matched
UNIQUE_OFFENSIVE_TERMS = list(set(OFFENSIVE_TERMS))

# --- Helper Functions (Normalization) ---

def normalize_hinglish_slang(text):
    """Cleans up common Hinglish phonetic substitutions, symbols, and repetitions."""
    text = text.lower()
    text = text.replace('!', 'i').replace('@', 'a').replace('$', 's')
    text = text.replace('0', 'o').replace('3', 'e').replace('5', 's')
    text = text.replace('7', 't').replace('1', 'l')
    text = re.sub(r'([a-z])\1{1,}', r'\1', text)
    return text

def remove_diacritics(text):
    """Removes accents/diacritics."""
    normalized_text = unicodedata.normalize('NFD', text)
    return "".join([c for c in normalized_text if not unicodedata.combining(c)])

def standardize_text(text):
    """Applies all normalization steps."""
    text = remove_diacritics(text)
    text = normalize_hinglish_slang(text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# --- Alignment Helper Function ---

def get_raw_index_map(raw_input):
    """Creates a mapping of normalized tokens back to their raw index span."""
    raw_index_map = []
    
    # Use regex to find non-space tokens (words) for accurate index tracking
    for match in re.finditer(r'\S+', raw_input):
        token = match.group(0)
        normalized_token = standardize_text(token)
        
        if normalized_token:
            raw_index_map.append({
                'norm_token': normalized_token,
                'raw_start': match.start(),
                'raw_end': match.end(),
                'matched': False
            })
            
    return raw_index_map

# --- Main Callable Function ---

def moderation_filter(raw_input_string):
    """
    Scans the raw input string for offensive words/phrases using normalization 
    and fuzzy matching, returning raw string indices.
    
    Args:
        raw_input_string (str): The text to be scanned.
        
    Returns:
        tuple: (filtered_string, list_of_matches, number_of_matches)
               - filtered_string (str): The fully normalized version of the input.
               - list_of_matches (list[tuple]): [(start_index, end_index, matched_term, match_type), ...]
               - number_of_matches (int): Total unique matches found.
    """
    normalized_text = standardize_text(raw_input_string)
    raw_map = get_raw_index_map(raw_input_string)
    matches = []
    
    sorted_terms = sorted(UNIQUE_OFFENSIVE_TERMS, key=len, reverse=True)

    # --- 1. Exact Match (Token-based for reliable indexing) ---
    
    for term in sorted_terms:
        # Only check single words or terms exactly matching a token
        if len(term.split()) == 1: 
            for entry in raw_map:
                if entry['norm_token'] == term and not entry['matched']:
                    matches.append((
                        entry['raw_start'],
                        entry['raw_end'],
                        term,
                        "Exact"
                    ))
                    entry['matched'] = True 

    # --- 2. Fuzzy Match (Fallback on remaining unmatched tokens) ---
    
    for entry in raw_map:
        if entry['matched']:
            continue
            
        input_word = entry['norm_token']
        if len(input_word) < 3: continue

        for term in UNIQUE_OFFENSIVE_TERMS:
            if len(term.split()) == 1:
                ratio = difflib.SequenceMatcher(None, input_word, term).ratio()
                
                if ratio >= SIMILARITY_THRESHOLD and ratio < 1.0:
                    matches.append((
                        entry['raw_start'],
                        entry['raw_end'],
                        term, 
                        f"Fuzzy ({ratio*100:.2f}%)"
                    ))
                    entry['matched'] = True
                    break 
    
    # Remove duplicate match entries (e.g., if multiple fuzzy matches led to the same index)
    unique_matches = list(set(matches))

    return normalized_text, unique_matches, len(unique_matches)