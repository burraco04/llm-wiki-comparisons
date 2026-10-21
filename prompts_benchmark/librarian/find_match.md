Determine whether the incoming note has the same subject as one existing note.

Choose a match only when both notes describe the same entity, event, concept or process.
Choose null when uncertain or merely related.

Return only one short reasoning sentence.

Incoming note:
<note>
{{note}}
</note>

Existing notes:
<notes>
{{notes}}
</notes>

Return exactly:
{
  "reasoning": "...",
  "match": "<exact path>" | null
}