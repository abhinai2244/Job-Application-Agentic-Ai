import re
from PyPDF2 import PdfReader

def read_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ""

    for page in reader.pages:
        text = page.extract_text() or ""
        raw_text += text

    # Aggressive cleaning
    clean_text = raw_text

    # Remove non-printable and non-ASCII characters
    clean_text = re.sub(r"[^\x20-\x7E\n]", "", clean_text)

    # Remove bullet points and common artifacts
    clean_text = re.sub(r"[•▪♦·●○‣⁃]", "", clean_text)

    # Remove all punctuation except periods and commas (optional)
    clean_text = re.sub(r"[^\w\s.,\n]", "", clean_text)

    # Normalize multiple spaces and newlines
    clean_text = re.sub(r"[ \t]+", " ", clean_text)       # multiple spaces -> single space
    clean_text = re.sub(r"\n{2,}", "\n\n", clean_text)     # multiple newlines -> 2 newlines
    clean_text = re.sub(r" +\n", "\n", clean_text)         # space before newline -> newline
    clean_text = clean_text.strip()

    # Collapse single-line sections that were likely headings
    clean_text = re.sub(r"\n(?=\S)", " ", clean_text)

    return clean_text
