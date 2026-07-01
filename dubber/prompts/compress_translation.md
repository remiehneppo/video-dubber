The Vietnamese translation is too long for the original timestamp.

Task:
- Shorten the Vietnamese text for natural TTS delivery.
- Preserve the source meaning, technical meaning, names, formulas, symbols, variables, units, and numbers.
- Preserve locked glossary terms that are required for this segment.
- Preserve every protected span using its required spoken form; never use a forbidden rendering.
- Remove redundancy and overly literal phrasing; do not add new facts.
- If safe shortening would remove technical meaning, keep the text unchanged and report that it still needs timing compression.
- Keep the style natural for spoken educational commentary.
- Return JSON only.

Input segment:
{segment}

Output:
{
  "segment_id": "...",
  "compressed_vi_text": "..."
}
