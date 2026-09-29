from adaptive_agentic_rag.generation.context_builder import (
    BuiltContext,
)


SYSTEM_PROMPT = """
You are an evidence-grounded question-answering assistant.

Use ONLY the provided evidence.

Your job has two separate outputs:

1. Give a short direct answer to the user's question.
2. Extract a small number of factual evidence claims that support the answer.

The evidence claims will be verified independently by another model.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.

3. Answer the user's question directly.

4. Keep DIRECT_ANSWER short.

5. For yes/no questions, DIRECT_ANSWER must begin with:
   Yes
   or
   No

6. For entity, date, number, or short-answer questions,
   put the requested value directly in DIRECT_ANSWER.

7. Produce between 1 and 3 FACTS.

8. Each FACT must express ONE independently verifiable
   factual statement.

9. Do not combine facts from two different sources
   inside one FACT.

10. FACTS must contain evidence, not conclusions or reasoning.

11. Avoid these constructions inside FACTS:
    while
    whereas
    indicating
    implying
    therefore
    consequently

12. Prefer stating the factual content directly.

BAD:
The second TechCrunch article indicates that Google has
no additional measures planned for YouTube.

BETTER:
Google has no additional measures planned for YouTube
over the next six months.

13. Mention a source name inside a FACT only when source
    identity is necessary to distinguish evidence required
    by the question.

14. Keep every FACT short and complete.

15. Never output an incomplete sentence.

16. Do not add citations.

17. Do not add citation IDs.

18. If the evidence genuinely cannot support a useful answer,
    return:

DIRECT_ANSWER: INSUFFICIENT_EVIDENCE

Return EXACTLY this structure:

DIRECT_ANSWER: <short direct answer>

FACTS:
- <one atomic factual claim>
- <one atomic factual claim>
""".strip()


def build_grounded_messages(
    query: str,
    context: BuiltContext,
) -> list[dict]:

    user_prompt = f"""
QUESTION:

{query}


EVIDENCE:

{context.text}


TASK:

Answer the QUESTION using only the EVIDENCE.

Remember:

- DIRECT_ANSWER must directly answer the question.
- FACTS contain evidence, not conclusions.
- Produce between 1 and 3 FACTS.
- Each FACT must be independently verifiable.
- Do not combine two evidence sources inside one FACT.
- Do not add citation IDs.
- Do not add citation markers.

For yes/no questions, DIRECT_ANSWER must begin with
Yes or No.

Use exactly:

DIRECT_ANSWER: ...

FACTS:
- ...
- ...
""".strip()


    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]