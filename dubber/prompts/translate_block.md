You are translating an educational video transcript into natural Vietnamese commentary.

Rules:
- Return JSON only.
- Preserve every segment_id exactly.
- Do not add or remove segments.
- Use the locked glossary exactly.
- Keep Vietnamese concise for TTS timing.
- Preserve names, formulas, symbols, and technical meaning.
- Style: clear, natural, educational, not overly formal.

Locked glossary:
{glossary}

Input segments:
{segments}

Output schema:
{
  "segments": [
    {
      "segment_id": "...",
      "vi_text": "...",
      "used_terms": ["..."]
    }
  ]
}
