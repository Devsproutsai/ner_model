from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from html import unescape
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import RedirectResponse


def remove_html_tags(input_string):
    # Define the pattern to match HTML tags and their content
    pattern = re.compile(r'<.*?>')
    # Remove HTML tags and their content from the input string
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string

def remove_special_characters_except_comma(input_string):
    pattern = r'[^a-zA-Z0-9 ,]'
    cleaned_string = re.sub(pattern, ' ', input_string)
    return cleaned_string


# Example text
def preprocess(text, key):
    # Decode HTML entities
    Decoded_string = unescape(text)
    decoded_string = remove_html_tags(Decoded_string)
    


    # Get the list of stop words for English
    stop_words = set(stopwords.words('english'))

    if key == 1: 
        # Remove special characters and punctuation
        clean_string = re.sub(r'[^a-zA-Z\s]', ' ', decoded_string)
        #print("sp chars replaced by space:", clean_string)
    elif key == 2:
        #remove special characters and punctuation except ","
        clean_string = remove_special_characters_except_comma(decoded_string)

    # Tokenize the text
    words = word_tokenize(clean_string)

    # Filter out the stop words
    filtered_sentence = [w for w in words if not w.lower() in stop_words]
    
    final_sentence = " ".join(filtered_sentence)
    return final_sentence

#creating class for payload

class RawStringInput(BaseModel):
    """Request body for the extract_entities endpoint."""
    raw_string: str
    
app = FastAPI(title="NER Model trained on education data", description="A simple API that extracts degree and specialisation entities from a raw string.", version="0.0.1")


def extract_degree_and_specialisation(model_dir, raw_string):
    """
    Load a custom spaCy NER model from a directory and extract degree and specialisation entities from a raw string.
    
    Args:
    - model_dir (str): Path to the directory containing the spaCy model.
    - raw_string (str): The raw input string.
    
    Returns:
    - result (dict): Dictionary with keys 'degree' and 'specialisation' and their extracted values.
    """
    # Load the spaCy model
    nlp = spacy.load(model_dir)
    
    # Process the raw string with the loaded model
    doc = nlp(preprocess(raw_string,2))
    result = {}
    
    # Extract entities
    for ent in doc.ents:
        if ent.label_=="DEGREE":
            result.setdefault("DEGREE",[]).append(ent.text)
        if ent.label_=="SPEC":
            result.setdefault("SPEC",[]).append(ent.text)
    
    return result
@app.get("/")
async def root():
    """Redirect to the API documentation page."""
    return RedirectResponse(url="/docs")
@app.post("/extract_entities")
async def extract_entities(input_: RawStringInput):
    """extract entities from the raw string"""
    model_directory = "./models/version1/model-best/"  # Update this with the actual path to your model
    result = extract_degree_and_specialisation(model_directory, input_.raw_string)
    return result
