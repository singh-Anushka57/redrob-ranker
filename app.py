#!/usr/bin/env python3
"""
app.py — HuggingFace Spaces Gradio sandbox
Accepts ≤100 candidate JSON records, ranks them, displays preview, and returns CSV.
"""
import json
import csv
import io
import tempfile
from datetime import date, datetime
from pathlib import Path
import gradio as gr
import numpy as np

# ----- Constants and Rules from pipeline.py -----
HARD_REQUIRED_SKILLS = [
    'embeddings','faiss','pinecone','weaviate','qdrant','milvus',
    'opensearch','elasticsearch','vector','retrieval','rag',
    'semantic search','hybrid search','ranking','reranking',
    'nlp','sentence transformers','bge','recommendation',
    'information retrieval','bm25','python','machine learning',
    'llm','transformers','fine-tuning','evaluation','ndcg',
    'haystack','mlflow','huggingface','pytorch','tensorflow',
]

RETRIEVAL_SKILLS = {
    'embedding','faiss','vector','retrieval','rag','semantic','hybrid','ranking',
    'reranking','recommendation','bm25','opensearch','elasticsearch','pinecone',
    'weaviate','qdrant','haystack','nlp','information retrieval','fine-tuning llms',
    'fine-tuning','sentence transformers','llm','transformers','machine learning',
}

CONSULTING = {
    'tcs','tata consultancy','infosys','wipro','accenture','cognizant',
    'capgemini','hcl','tech mahindra','mphasis','hexaware','mindtree',
    'ltimindtree','ibm global','cts','l&t infotech'
}

BAD_TITLES = {
    'hr manager','operations manager','marketing manager','content writer',
    'accountant','civil engineer','mechanical engineer','customer support',
    'business analyst','project manager','sales manager','qa engineer',
    'financial analyst','supply chain','graphic designer'
}

HARD_REQUIRED_GEN = {
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "opensearch",
    "elasticsearch", "vector", "retrieval", "rag", "semantic search",
    "ranking", "nlp", "sentence transformers", "bge", "hybrid search",
    "recommendation", "information retrieval", "reranking", "bm25", "python",
    "machine learning", "llm", "transformers", "fine-tuning", "ndcg", "evaluation",
    "a/b testing"
}

# =====================================================================
# SCORING HELPERS
# =====================================================================

def skill_score(skills) -> float:
    if not skills:
        return 0.0
    names = ' '.join(s.get('name', '').lower() for s in skills)
    matches = sum(1 for r in HARD_REQUIRED_SKILLS if r in names)
    advanced_bonus = sum(
        0.5 for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
        and any(r in s.get('name', '').lower() for r in RETRIEVAL_SKILLS)
    )
    return min((matches + advanced_bonus) / 10.0, 1.0)


def exp_score(yoe: float, company_type: str) -> float:
    if yoe < 2:
        base = 0.1
    elif yoe < 4:
        base = 0.4
    elif yoe <= 9:
        base = 0.7 + (min(yoe, 8) - 4) / 4 * 0.3
    elif yoe <= 12:
        base = 0.9
    else:
        base = 0.7
    bonus = {'product': 0.15, 'mixed': 0.05, 'consulting_only': -0.30}
    return max(0.0, min(1.0, base + bonus.get(company_type, 0.0)))


def company_type_fn(career) -> str:
    companies = [h.get("company", "") for h in career]
    consulting_flags = [any(c in cn.lower() for c in CONSULTING) for cn in companies]
    if not consulting_flags:
        return "product"
    if all(consulting_flags):
        return "consulting_only"
    if any(consulting_flags):
        return "mixed"
    return "product"


def title_score_fn(title: str) -> float:
    tl = title.lower()
    if any(bad in tl for bad in BAD_TITLES):
        return 0.0
    good = [
        'machine learning', 'ml engineer', 'ai engineer', 'nlp',
        'search engineer', 'data scientist', 'applied scientist',
        'research engineer', 'recommendation', 'retrieval',
        'ranking engineer', 'principal engineer', 'staff engineer',
    ]
    if any(g in tl for g in good):
        return 1.0
    if any(k in tl for k in ['engineer', 'developer', 'scientist', 'architect']):
        return 0.6
    return 0.2


def location_score_fn(loc: str, country: str, rel: bool) -> float:
    preferred = {'noida','pune','hyderabad','mumbai','delhi','gurgaon','gurugram','bangalore','bengaluru','ncr'}
    if any(p in loc.lower() for p in preferred):
        return 1.0
    if country == 'India':
        return 0.85 if rel else 0.6
    return 0.5 if rel else 0.2


def disqualifier_penalty(title: str, company_type: str) -> float:
    tl = title.lower()
    if any(bad in tl for bad in BAD_TITLES):
        return 0.05
    if company_type == 'consulting_only':
        return 0.25
    return 1.0


def compute_hire_prob(sig) -> float:
    # 1. Responsiveness
    rrr = sig.get("recruiter_response_rate", 0.5)
    resp_time = sig.get("avg_response_time_hours", 24)
    responsiveness = rrr * (1.0 / np.log1p(resp_time + 1.0))
    
    # 2. Interview Intent
    icr = sig.get("interview_completion_rate", 0.5)
    open_to_work = sig.get("open_to_work_flag", False)
    open_to_work_bonus = 1.2 if open_to_work else 1.0
    interview_intent = icr * open_to_work_bonus
    
    # 3. Offer Intent
    oar = sig.get("offer_acceptance_rate", -1)
    offer_intent = oar if oar >= 0.0 else 0.5
    
    # 4. Recency Factor
    try:
        last_active = datetime.strptime(sig["last_active_date"], "%Y-%m-%d").date()
        days = (date.today() - last_active).days
    except Exception:
        days = 999
    recency_factor = np.exp(-days * np.log(2.0) / 45.0)
    
    # 5. Notice Factor
    notice = sig.get("notice_period_days", 60)
    if notice <= 30:
        notice_factor = 1.0
    elif notice <= 60:
        notice_factor = 0.85
    elif notice <= 90:
        notice_factor = 0.70
    else:
        notice_factor = 0.55
        
    hire_prob = (
        responsiveness * 0.30 +
        interview_intent * 0.30 +
        offer_intent * 0.20 +
        recency_factor * 0.20
    ) * notice_factor
    return float(np.clip(hire_prob, 0.0, 1.0))


def extract_facts(cand: dict) -> dict:
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])

    hard_matches = [s["name"] for s in skills if any(h in s["name"].lower() for h in HARD_REQUIRED_GEN)]
    product_companies = [
        h["company"] for h in career
        if not any(c in h["company"].lower() for c in CONSULTING)
    ]
    yoe = p.get("years_of_experience", 0) or 0
    title = p.get("current_title", "")
    
    assessment = sig.get("skill_assessment_scores", {}) or {}
    relevant_assessments = [
        f"{k.capitalize()} ({v}/100)" for k, v in assessment.items()
        if v is not None and (k.lower() in HARD_REQUIRED_GEN or any(h in k.lower() for h in HARD_REQUIRED_GEN))
    ]

    last_active_str = sig.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (date.today() - last_active).days
    except Exception:
        days_inactive = 999

    notice_days = sig.get("notice_period_days", 60)
    open_to_work = sig.get("open_to_work_flag", False)

    return {
        "title": title,
        "company": p.get("current_company", ""),
        "yoe": yoe,
        "location": p.get("location", ""),
        "hard_matches": hard_matches,
        "product_companies": product_companies,
        "relevant_assessments": relevant_assessments,
        "days_inactive": days_inactive,
        "open_to_work": open_to_work,
        "notice_days": notice_days,
        "github_score": sig.get("github_activity_score", -1),
    }


def build_reasoning(facts: dict, cid: str) -> str:
    f = facts
    title = f["title"] if f["title"] else "AI Engineer"
    
    company = "a product company"
    if f["product_companies"]:
        company = f["product_companies"][0]
    elif f["company"]:
        company = f["company"]
        
    skills = "machine learning"
    if f["hard_matches"]:
        skills = ", ".join(f["hard_matches"][:3])
        
    notice = f"{f['notice_days']}-day" if f["notice_days"] > 0 else "immediate"
    if f["notice_days"] <= 30:
        notice = "short"
        
    work_status = "actively open to work" if f["open_to_work"] else "receptive to opportunities"
    
    days_active = f"last active {f['days_inactive']} days ago" if f["days_inactive"] != 999 else "active on the platform"
    if f["days_inactive"] <= 14:
        days_active = "highly active recently"
        
    location = f["location"] if f["location"] else "India"
    
    assessments_str = ""
    if f["relevant_assessments"]:
        assessments_str = f"with scores like {', '.join(f['relevant_assessments'][:2])}"
    else:
        assessments_str = f"demonstrated via key skills"
        
    github_str = f"with a GitHub score of {f['github_score']:.0f}" if f["github_score"] >= 0 else "having a complete profile"

    template_idx = abs(hash(cid)) % 8
    
    if template_idx == 0:
        s1 = f"Possesses {f['yoe']:.1f} years of experience as a {title}, with key expertise in {skills}."
        s2 = f"Currently associated with {company}, they are {work_status} with a {f['notice_days']}-day notice period."
    elif template_idx == 1:
        s1 = f"An experienced {title} with a career history including {company} and {f['yoe']:.1f} total years of experience."
        s2 = f"They possess strong skill matches in {skills} and maintain a {notice} notice period."
    elif template_idx == 2:
        s1 = f"Having worked as a {title} at {company}, this candidate has accumulated {f['yoe']:.1f} years of experience."
        s2 = f"Their profile highlights expertise in {skills} {assessments_str} while being {days_active}."
    elif template_idx == 3:
        s1 = f"Demonstrates strong proficiency in {skills} from {f['yoe']:.1f} years in roles like {title}."
        s2 = f"Based in {location}, they are {work_status} with a {f['notice_days']}-day notice period."
    elif template_idx == 4:
        s1 = f"With {f['yoe']:.1f} years of experience, this candidate has served as a {title} at {company}."
        s2 = f"Their technical skill set includes {skills} {assessments_str}, and they show a {notice} notice period."
    elif template_idx == 5:
        s1 = f"As a {title} with {f['yoe']:.1f} years of experience, they have developed hands-on skills in {skills}."
        s2 = f"They are currently {work_status} {github_str} and are {days_active}."
    elif template_idx == 6:
        s1 = f"This candidate has spent {f['yoe']:.1f} years in the industry, most recently as a {title} at {company}."
        s2 = f"They exhibit strong capabilities in {skills} and can start within a {f['notice_days']}-day notice period."
    else:  # template_idx == 7
        s1 = f"A specialist in {skills} who has spent {f['yoe']:.1f} years of experience in roles including {title}."
        s2 = f"Their background at {company} is supported by their {work_status} status and being {days_active}."
        
    return f"{s1} {s2}"


# =====================================================================
# CANDIDATE SCORER (SANDBOX TF-IDF PROXY)
# =====================================================================

def score_candidate(cand) -> float:
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    
    ctype = company_type_fn(career)
    
    # Sub-scores
    sk = skill_score(skills)
    ex = exp_score(p.get("years_of_experience", 0), ctype)
    ts = title_score_fn(p.get("current_title", ""))
    ls = location_score_fn(p.get("location", ""), p.get("country", ""), sig.get("willing_to_relocate", False))
    
    # TF-IDF Proxy
    semantic = sk * 0.6 + ex * 0.4
    
    # Fit Score
    fit_score = (
        0.50 * semantic +
        0.20 * sk +
        0.15 * ex +
        0.08 * ts +
        0.07 * ls
    )
    
    # Hire Probability
    hire_prob = compute_hire_prob(sig)
    final_score = fit_score * (0.75 + 0.25 * hire_prob)
    
    # Penalties
    disq = disqualifier_penalty(p.get("current_title", ""), ctype)
    
    # Honeypot detection
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    assessment = sig.get("skill_assessment_scores", {}) or {}
    assessment_expert_count = sum(1 for v in assessment.values() if v >= 88)
    yoe = p.get("years_of_experience", 0) or 0
    total_tenure_months = sum(h.get("duration_months", 0) for h in career)
    tenure_inflation = total_tenure_months > (yoe * 12 + 30)
    
    is_honeypot = expert_count >= 8 or assessment_expert_count >= 5 or tenure_inflation
    honeypot_penalty = 0.01 if is_honeypot else 1.0
    
    final_score = final_score * disq * honeypot_penalty
    return float(np.clip(final_score, 0.0, 1.0))


# =====================================================================
# GRADIO HANDLER
# =====================================================================

def rank_candidates(file_upload, json_text):
    try:
        if file_upload is not None:
            content = file_upload.decode("utf-8")
        elif json_text.strip():
            content = json_text.strip()
        else:
            return None, "Please upload a JSON file or paste JSON content.", []

        # Parse JSON array or JSONL
        candidates = []
        try:
            data = json.loads(content)
            candidates = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))

        if len(candidates) > 100:
            return None, f"Sandbox limited to 100 candidates; got {len(candidates)}.", []

        # Score and Sort
        scored = []
        for cand in candidates:
            try:
                score = score_candidate(cand)
                scored.append((cand, round(score, 6)))
            except Exception:
                pass

        # Sort: score descending, candidate_id ascending for tie-break
        scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))
        
        # Build CSV and Table Data
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        table_data = []
        prev_score = None
        for rank, (cand, score) in enumerate(scored, start=1):
            cid = cand.get("candidate_id", f"CAND_{rank:07d}")
            score_r = round(score, 6)
            
            # Non-increasing score
            if prev_score is not None and score_r > prev_score:
                score_r = prev_score
            prev_score = score_r
            
            facts = extract_facts(cand)
            reasoning = build_reasoning(facts, cid)
            
            writer.writerow([cid, rank, score_r, reasoning])
            table_data.append([
                rank,
                cid,
                score_r,
                cand.get("profile", {}).get("current_title", ""),
                cand.get("profile", {}).get("location", ""),
                reasoning
            ])

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8")
        tmp.write(output.getvalue())
        tmp.close()

        return tmp.name, f"Ranked {len(scored)} candidates successfully. Download the CSV below.", table_data

    except Exception as e:
        return None, f"Error processing candidates: {str(e)}", []


# =====================================================================
# UI LAYOUT
# =====================================================================

with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown("# Redrob Candidate Ranker (Option C Sandbox)")
    gr.Markdown("Upload up to 100 candidate records (JSON array or JSONL). Rankings use the TF-IDF proxy online but match offline composite logic.")

    with gr.Row():
        file_in = gr.File(label="Upload candidates.json / .jsonl", file_types=[".json", ".jsonl"])
        text_in = gr.Textbox(label="Or paste JSON/JSONL array here", lines=8)

    run_btn = gr.Button("Rank Candidates", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    file_out = gr.File(label="Download Ranked CSV")
    
    table_out = gr.Dataframe(
        headers=["Rank", "Candidate ID", "Score", "Title", "Location", "Reasoning"],
        datatype=["number", "str", "number", "str", "str", "str"],
        label="Ranked Table Preview"
    )

    run_btn.click(
        fn=rank_candidates,
        inputs=[file_in, text_in],
        outputs=[file_out, status, table_out]
    )

if __name__ == "__main__":
    demo.launch()
