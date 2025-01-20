# Basic constants placeholder
class Constants:
    # Add your constants here
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    ERROR = "ERROR"
    
    # HTTP Status codes
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_400_BAD_REQUEST = 400
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404
    HTTP_500_INTERNAL_SERVER_ERROR = 500

# Constants specifically for LLM operations
LANGUAGE_MODEL_NAMES = [
    "gpt-3.5-turbo",
    "gpt-4",
    "claude-2"
]

MISTRALAI_LANGUAGE_MODEL_NAMES = [
    "mistral-tiny",
    "mistral-small",
    "mistral-medium",
    "mistral-large"
]

# Data directory constants
SYNTH_DIFFUSE_DATA_DIR = "data/synthetic_diffusion"

SYNTH_ONLY_CATEGORIES = ["relation", "counting"]

# COCO dataset constants
TOTAL_NUM_COCO_CHUNKS = 10

# Valid categories
VALID_CATEGORY_NAMES = [
    "person", "vehicle", "outdoor", "animal",
    "accessory", "sports", "kitchen", "food",
    "furniture", "electronic", "appliance", "indoor"
] 

VALID_SPATIAL_DIRECTIONS = ["left", "right", "top", "bottom", "below", "above", "under"]

import json 
def load_json_data(filepath: str) -> dict:
    """
    Load data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        dict: Loaded JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded JSON data from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {str(e)}")
        raise

def save_to_json(data: dict, filepath: str, indent: int = 4) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Path where to save the JSON file
        indent (int, optional): Number of spaces for indentation. Defaults to 4.
        
    Raises:
        TypeError: If data is not JSON serializable
        IOError: If there's an error writing to the file
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"Successfully saved JSON data to {filepath}")
    except TypeError as e:
        logger.error(f"Data is not JSON serializable: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"Error writing to file {filepath}: {str(e)}")
        raise
