import gradio as gr
import requests
import os
import re
import spacy
import spacy.cli
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util

# Ensure spaCy model is available
try:
    spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Detect ATS from URL
def detect_ats_platform(url):
    ats_map = {
        "greenhouse.io": "Greenhouse",
        "myworkdayjobs.com": "Workday",
        "icims.com": "iCIMS",
        "taleo.net": "Taleo",
        "lever.co": "Lever",
        "smartrecruiters.com": "SmartRecruiters",
        "jobvite.com": "Jobvite",
        "adp.com": "ADP",
        "successfactors.com": "SuccessFactors",
        "brassring.com": "BrassRing",
        "jazzhr.com": "JazzHR",
        "breezy.hr": "BreezyHR",
        "jobdiva.com": "JobDiva",
        "bullhorn.com": "Bullhorn",
        "bamboohr.com": "BambooHR"
    }
    for domain, name in ats_map.items():
        if domain in url:
            return name
    return "Unknown"

# Extract job description from page
def get_job_description(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        blocks = [tag.get_text(separator="\n").strip()
                  for tag in soup.find_all(['div', 'section'])
                  if len(tag.get_text()) > 500]
        return max(blocks, key=len) if blocks else "❌ No job description found."
    except Exception as e:
        return f"❌ Error: {e}"

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()

def semantic_score(resume_text, jd_text):
    embeddings = model.encode([resume_text, jd_text])
    return float(util.cos_sim(embeddings[0], embeddings[1]))

def keyword_score(resume_text, jd_text):
    vectorizer = CountVectorizer(stop_words='english').fit([jd_text, resume_text])
    vectors = vectorizer.transform([jd_text, resume_text])
    keywords = vectorizer.get_feature_names_out()
    jd_vector = vectors[0].toarray().flatten()
    resume_vector = vectors[1].toarray().flatten()
    matched = [kw for i, kw in enumerate(keywords) if jd_vector[i] > 0 and resume_vector[i] > 0]
    return matched, len(matched) / max(len(set(keywords)), 1)

def compute_score(resume_text, jd_text, platform):
    sim_weight, kw_weight = 0.6, 0.4
    if platform in ["Workday", "Taleo", "SuccessFactors", "ADP"]:
        sim_weight, kw_weight = 0.5, 0.5
    elif platform in ["Lever", "JazzHR", "SmartRecruiters"]:
        sim_weight, kw_weight = 0.7, 0.3

    sim = semantic_score(resume_text, jd_text)
    matched_keywords, kw = keyword_score(resume_text, jd_text)
    score = round((sim * sim_weight + kw * kw_weight) * 100)
    return score, matched_keywords

# Core pipeline
def match_resume(file, url):
    if file is None or url.strip() == "":
        return "Please upload a PDF and paste a job URL.", "", ""

    resume_text = extract_text(file.name)
    resume_clean = clean_text(resume_text)
    jd_text = get_job_description(url)
    if jd_text.startswith("❌"):
        return jd_text, "", ""

    jd_clean = clean_text(jd_text)
    platform = detect_ats_platform(url)
    score, matched_keywords = compute_score(resume_clean, jd_clean, platform)

    summary = f"✅ ATS: {platform}\n🎯 Score: {score}/100\n🔑 Matched Keywords: {', '.join(matched_keywords)}"
    if score < 80:
        summary += "\n⚠️ Consider adding more relevant terms and phrasing from the job post."
    else:
        summary += "\n🚀 Great alignment!"

    return jd_text, score, summary

# Gradio UI
iface = gr.Interface(
    fn=match_resume,
    inputs=[
        gr.File(label="Upload Resume (PDF)", file_types=[".pdf"]),
        gr.Textbox(label="Paste Job URL")
    ],
    outputs=[
        gr.Textbox(label="Extracted Job Description", lines=10),
        gr.Number(label="Match Score"),
        gr.Textbox(label="Summary & Keyword Match", lines=8)
    ],
    title="📄 Resume Matcher with Gradio",
    description="Match your resume to any job description from Greenhouse, Lever, Workday, and more.",
)

if __name__ == "__main__":
    iface.launch()
