import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

ALLOWED_CLASSES = ["10"]
ALLOWED_SUBJECTS = ["Maths", "Physics"]

CHAPTERS = {
    "Maths": [
        "Commercial Mathematics",
        "Algebra",
        "Geometry",
        "Mensuration",
        "Trigonometry",
    ],
    "Physics": [
        "Force, Work, Power and Energy",
        "Light",
        "Sound",
        "Electricity and Magnetism",
        "Heat",
        "Modern Physics",
    ],
}
