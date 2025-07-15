import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
import re
import spacy

# Safe initialization
try:
    spacy.cli.download("en_core_web_sm")
except:
    pass

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')


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


def get_job_description(url):
    platform = detect_ats_platform(url)
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        if platform == "Greenhouse":
            tag = soup.find('div', class_='content')
            if tag:
                return tag.get_text(separator="\n").strip()
            blocks = [t.get_text(separator="\n").strip() for t in soup.find_all(['div', 'section']) if len(t.get_text()) > 500]
            return max(blocks, key=len) if blocks else "❌ No readable job description found."

        mapping = {
            "Workday": ('div', 'css-1yqjbmw'),
            "iCIMS": ('div', 'iCIMS_JobContent'),
            "Taleo": ('div', 'requisitionDescriptionInterface.ID1527.row1'),
            "Lever": ('div', 'section page-centered'),
            "SmartRecruiters": ('div', 'description-section'),
            "Jobvite": ('div', 'job-description'),
            "ADP": ('div', 'jdp-body'),
            "SuccessFactors": ('div', 'jobdescription'),
            "BrassRing": ('div', 'jobdescription'),
            "JazzHR": ('div', 'jobDescription'),
            "BreezyHR": ('div', 'posting-section'),
            "JobDiva": ('div', 'job_description'),
            "Bullhorn": ('div', 'bh-job-description'),
            "BambooHR": ('div', 'ats-description')
        }

        tag_name, tag_class = mapping.get(platform, (None, None))
        if tag_name and tag_class:
            tag = soup.find(tag_name, class_=tag_class) or soup.find(tag_name, id=tag_class)
            if tag:
                return tag.get_text(separator="\n").strip()
        return f"❌ Could not find job description for {platform}."

    except Exception as e:
        return f"❌ Error fetching job description: {e}"


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


def run_app():
    st.set_page_config(page_title="Resume Matcher", layout="centered")
    st.title("📄 Resume–Job Matcher (Multi-ATS)")
    st.markdown("Upload your resume and paste a job URL from Greenhouse, Lever, Workday, and more.")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_url = st.text_input("Paste Job URL from an ATS platform")

    if uploaded_file and job_url:
        with st.spinner("🔍 Analyzing your resume..."):
            resume_text = extract_text(uploaded_file)
            resume_clean = clean_text(resume_text)

            jd_text = get_job_description(job_url)
            if jd_text.startswith("❌"):
                st.error(jd_text)
            else:
                jd_clean = clean_text(jd_text)
                platform = detect_ats_platform(job_url)
                score, matched_keywords = compute_score(resume_clean, jd_clean, platform)

                st.success(f"🎯 Match Score: {score}/100")
                st.markdown(f"**✅ ATS Detected:** {platform}")
                st.markdown(f"**🔑 Matched Keywords ({len(matched_keywords)}):**")
                st.code(", ".join(matched_keywords))

                if score < 80:
                    st.warning("⚠️ Suggestions to Improve Your Resume:")
                    st.markdown("- Add more relevant keywords from the job post")
                    st.markdown("- Mirror the phrasing used in the job description")
                    st.markdown("- Highlight tools, KPIs, and quantifiable results")
                else:
                    st.success("🚀 Excellent match! Your resume aligns well with this role.")


if __name__ == "__main__":
    run_app()
