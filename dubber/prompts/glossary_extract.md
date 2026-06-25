You are extracting domain-specific terminology from an English educational video transcript.

Domain: {domain}

Task:
1. Extract important domain terms, proper nouns, abbreviations, formulas, and phrases.
2. Suggest concise Vietnamese translations.
3. Keep technical terms consistent with Vietnamese academic usage.
4. Return JSON only.

Output schema:
{
  "terms": [
    {
      "original": "...",
      "vietnamese": "...",
      "category": "math_term | proper_noun | abbreviation | phrase | formula",
      "confidence": 0.0,
      "source_segments": ["seg_000001"],
      "notes": ""
    }
  ]
}
