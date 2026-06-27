#!/usr/bin/env python3
import json, csv
from datetime import date, datetime
from pathlib import Path

HARD_REQUIRED = [
    'embeddings','faiss','pinecone','weaviate','qdrant','milvus',
    'opensearch','elasticsearch','vector','retrieval','rag',
    'semantic search','hybrid search','ranking','reranking',
    'nlp','sentence transformers','bge','recommendation',
    'information retrieval','bm25','python','machine learning',
    'llm','transformers','fine-tuning','evaluation','ndcg',
    'haystack','mlflow','huggingface','pytorch','tensorflow',
]

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

PREFERRED_LOCS = {
    'noida','pune','hyderabad','mumbai','delhi','gurgaon','gurugram',
    'bangalore','bengaluru','ncr'
}

JD_TERMS = set([
    'embedding','embeddings','faiss','vector','retrieval','rag','semantic',
    'hybrid','ranking','reranking','nlp','transformers','bge','e5',
    'recommendation','information','bm25','python','machine','learning','llm',
    'fine','tuning','evaluation','ndcg','search','pinecone','weaviate','qdrant',
    'opensearch','elasticsearch','milvus','haystack','huggingface','pytorch',
    'tensorflow','mlflow','xgboost','testing','pipeline','inference','production',
    'deployed','shipped','retrieval-augmented','augmented','rerank','sentence',
])

RETRIEVAL_SKILLS = {
    'embedding','faiss','vector','retrieval','rag','semantic','hybrid','ranking',
    'reranking','recommendation','bm25','opensearch','elasticsearch','pinecone',
    'weaviate','qdrant','haystack','nlp','information retrieval','fine-tuning llms',
    'fine-tuning','sentence transformers','llm','transformers','machine learning',
}


def company_type(career):
    flags = [any(c in h['company'].lower() for c in CONSULTING) for h in career]
    if not flags:
        return 'product'
    if all(flags):
        return 'consulting_only'
    if any(flags):
        return 'mixed'
    return 'product'


def skill_score(skills):
    names = ' '.join(s['name'].lower() for s in skills)
    matches = sum(1 for r in HARD_REQUIRED if r in names)
    # Also weight by proficiency
    advanced_bonus = sum(
        0.5 for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
        and any(r in s['name'].lower() for r in RETRIEVAL_SKILLS)
    )
    return min((matches + advanced_bonus) / 10.0, 1.0)


def semantic_score(cand):
    p = cand['profile']
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    parts = [
        p.get('headline', ''),
        p.get('summary', '')[:600],
        ' '.join(s['name'] for s in skills),
    ]
    for h in career[:3]:
        parts.append(h.get('title', ''))
        parts.append(h.get('description', '')[:400])
    text = ' '.join(parts).lower()
    words = set(text.replace(',', ' ').replace('.', ' ').replace('-', ' ').split())
    overlap = len(words & JD_TERMS)
    return min(overlap / 12.0, 1.0)


def exp_score(yoe, ctype):
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
    return max(0.0, min(1.0, base + bonus.get(ctype, 0)))


def title_sc(title):
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


def loc_sc(loc, country, rel):
    if any(p in loc.lower() for p in PREFERRED_LOCS):
        return 1.0
    if country == 'India':
        return 0.85 if rel else 0.6
    return 0.5 if rel else 0.2


def avail_mult(sig):
    m = 1.0
    try:
        last = datetime.strptime(sig['last_active_date'], '%Y-%m-%d').date()
        days = (date.today() - last).days
        if days <= 14:
            m *= 1.15
        elif days <= 30:
            m *= 1.08
        elif days <= 60:
            m *= 1.0
        elif days <= 90:
            m *= 0.92
        elif days <= 180:
            m *= 0.80
        else:
            m *= 0.60
    except Exception:
        m *= 0.80

    m *= 1.12 if sig.get('open_to_work_flag') else 0.90
    rrr = sig.get('recruiter_response_rate', 0.5)
    m *= (0.7 + rrr * 0.6)
    notice = sig.get('notice_period_days', 60)
    if notice <= 30:
        m *= 1.05
    elif notice <= 60:
        m *= 1.0
    elif notice <= 90:
        m *= 0.93
    else:
        m *= 0.82
    icr = sig.get('interview_completion_rate', 0.5)
    m *= (0.8 + icr * 0.4)
    return float(min(max(m, 0.35), 1.35))


def disq(title, ctype, honeypot=False):
    if honeypot:
        return 0.01
    tl = title.lower()
    if any(bad in tl for bad in BAD_TITLES):
        return 0.05
    if ctype == 'consulting_only':
        return 0.25
    return 1.0


def is_honeypot(cand):
    p = cand['profile']
    sig = cand['redrob_signals']
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    yoe = p.get('years_of_experience', 0) or 0
    expert = sum(1 for s in skills if s.get('proficiency') == 'expert')
    if expert >= 8:
        return True
    assessment = sig.get('skill_assessment_scores', {}) or {}
    if sum(1 for v in assessment.values() if v >= 88) >= 5:
        return True
    total_months = sum(h.get('duration_months', 0) for h in career)
    if total_months > (yoe * 12 + 36):
        return True
    # Perfect completeness but same-day signup + last_active
    try:
        if sig.get('profile_completeness_score', 0) >= 99:
            last = datetime.strptime(sig['last_active_date'], '%Y-%m-%d').date()
            signup = datetime.strptime(sig['signup_date'], '%Y-%m-%d').date()
            if (last - signup).days <= 1:
                return True
    except Exception:
        pass
    return False


def build_reasoning(cand):
    p = cand['profile']
    sig = cand['redrob_signals']
    skills = cand.get('skills', [])
    career = cand.get('career_history', [])
    hard_hits = [
        s['name'] for s in skills
        if any(h in s['name'].lower() for h in [
            'embedding', 'faiss', 'vector', 'retrieval', 'rag', 'nlp', 'ranking',
            'semantic', 'recommendation', 'opensearch', 'elasticsearch', 'pinecone',
            'weaviate', 'bm25', 'transformers', 'machine learning', 'llm',
            'fine-tuning', 'haystack', 'rerank',
        ])
    ]
    product_cos = [
        h['company'] for h in career
        if not any(c in h['company'].lower() for c in CONSULTING)
    ]
    yoe = p.get('years_of_experience', 0)
    title = p.get('current_title', '')
    company = p.get('current_company', '')

    if hard_hits and product_cos:
        s1 = "{:.1f}-year {} at {} with hands-on experience in {}.".format(
            yoe, title, product_cos[0], ', '.join(hard_hits[:3]))
    elif hard_hits:
        s1 = "{:.1f}-year {} with skills in {}.".format(
            yoe, title, ', '.join(hard_hits[:3]))
    else:
        s1 = "{:.1f}-year {} at {}; limited direct IR/ML skill match.".format(
            yoe, title, company)

    notes = []
    ctype = company_type(career)
    if ctype == 'consulting_only':
        notes.append('consulting-only career history')
    try:
        days = (date.today() - datetime.strptime(
            sig['last_active_date'], '%Y-%m-%d').date()).days
        if days > 180:
            notes.append('inactive {} days'.format(days))
        elif sig.get('open_to_work_flag'):
            notes.append('actively open to work')
    except Exception:
        pass
    nd = sig.get('notice_period_days', 60)
    if nd <= 15:
        notes.append('available immediately (notice {}d)'.format(nd))
    elif nd > 90:
        notes.append('long notice period ({}d)'.format(nd))

    s2 = ('; '.join(notes).capitalize() + '.') if notes else ''
    return (s1 + ' ' + s2).strip()


def main():
    print('Scoring 100,000 candidates ...')
    results = []
    with open('/mnt/user-data/uploads/candidates.jsonl') as f:
        for i, line in enumerate(f):
            if i % 20000 == 0:
                print('  {}/100000 ...'.format(i))
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            p = cand['profile']
            sig = cand['redrob_signals']
            career = cand.get('career_history', [])
            ctype = company_type(career)
            hp = is_honeypot(cand)
            sem = semantic_score(cand)
            sk = skill_score(cand.get('skills', []))
            ex = exp_score(p.get('years_of_experience', 0), ctype)
            ts = title_sc(p.get('current_title', ''))
            ls = loc_sc(p.get('location', ''), p.get('country', ''),
                        sig.get('willing_to_relocate', False))
            av = avail_mult(sig)
            dq = disq(p.get('current_title', ''), ctype, hp)
            final = (0.35 * sem + 0.25 * sk + 0.20 * ex + 0.10 * ts + 0.10 * ls) * av * dq
            final = min(max(final, 0.0), 1.0)
            results.append((final, cand))

    print('Sorting ...')
    results.sort(key=lambda x: -x[0])
    top100 = results[:100]

    hp_in_top = sum(1 for s, c in top100 if is_honeypot(c))
    print('Honeypots in top 100: {}/100'.format(hp_in_top))
    if hp_in_top > 10:
        print('WARNING: honeypot rate > 10% - check scoring!')

    out = '/mnt/user-data/outputs/team_submission.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        prev_score = None
        for rank, (score, cand) in enumerate(top100, start=1):
            score_r = round(score, 6)
            if prev_score is not None and score_r > prev_score:
                score_r = prev_score
            prev_score = score_r
            reasoning = build_reasoning(cand)
            writer.writerow([cand['candidate_id'], rank, score_r, reasoning])

    print('\nDone! Saved to team_submission.csv')
    print('\nTop 15 candidates:')
    for rank, (score, cand) in enumerate(top100[:15], 1):
        p = cand['profile']
        print('  #{:2d}  {}  {:.4f}  {:<38}  {}'.format(
            rank, cand['candidate_id'], score,
            p['current_title'][:38], p['location']))


if __name__ == '__main__':
    main()
