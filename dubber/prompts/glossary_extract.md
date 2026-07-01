You extract a compact locked glossary for an English educational video transcript.

Domain: {domain}
Domain profile: {domain_profile}
Domain guidance: {domain_profile_summary}

Rules:
- Return JSON only.
- Extract only terms that materially affect translation consistency: domain terms, proper nouns, abbreviations, formulas, symbols, units, and repeated key phrases.
- Do not include generic filler words or ordinary phrases unless they have a domain-specific meaning.
- Suggest concise natural Vietnamese suitable for spoken educational commentary.
- Preserve names, formulas, symbols, variables, and conventional notation exactly when translation would be harmful.
- Treat protected_spans supplied by the deterministic domain profile as authoritative; do not redefine or translate them.
- Cite source_segments using only segment_id values present in the input.
- Prefer fewer high-value terms over a long noisy glossary.

Protected spans:
{protected_spans}

Output schema:
{
  "terms": [
    {
      "term_id": "...",
      "original": "...",
      "vietnamese": "...",
      "category": "math_term | proper_noun | abbreviation | phrase | formula | symbol | unit",
      "confidence": 0.0,
      "source_segments": ["seg_000001"],
      "notes": ""
    }
  ]
}
