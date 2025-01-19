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

# COCO dataset constants
TOTAL_NUM_COCO_CHUNKS = 10

# Valid categories
VALID_CATEGORY_NAMES = [
    "person", "vehicle", "outdoor", "animal",
    "accessory", "sports", "kitchen", "food",
    "furniture", "electronic", "appliance", "indoor"
] 