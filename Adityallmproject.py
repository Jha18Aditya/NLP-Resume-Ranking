
import os
import pandas as pd
import spacy
from pdfminer.high_level import extract_text

# Load an optimized NLP model
nlp = spacy.load("en_core_web_lg")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with error handling."""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def calculate_match_score(resume_text, job_doc):
    """Calculate match score based on similarity, handling empty documents."""
    resume_doc = nlp(resume_text)
    return resume_doc.similarity(job_doc) if resume_text.strip() else 0

def process_resumes(resume_folder, job_description):
    """Process all resumes in a folder and rank them."""
    scores = []
    job_doc = nlp(job_description)

    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(resume_folder, filename)
            text = extract_text_from_pdf(file_path)
            score = calculate_match_score(text, job_doc)
            scores.append((filename, score))

    # Rank resumes by score
    ranked_resumes = sorted(scores, key=lambda x: x[1], reverse=True)
    results_df = pd.DataFrame(ranked_resumes, columns=["Resume", "Score"])
    
    # Save results to CSV
    results_df.to_csv("ranked_resumes.csv", index=False)
    return results_df

if __name__ == "__main__":
    folder_path = "resumes"  # Modify to actual folder path
    job_desc = "We are looking for a Python developer with experience in machine learning and NLP."
    
    results = process_resumes(folder_path, job_desc)
    print(results)
