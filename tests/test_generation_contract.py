from types import SimpleNamespace

from adaptive_agentic_rag.generation.prompts import (
    SYSTEM_PROMPT,
    build_grounded_messages,
)


def test_generation_prompt_keeps_direct_answer_before_facts():

    assert (
        "DIRECT_ANSWER: <short direct answer>\n\nFACTS:"
        in
        SYSTEM_PROMPT
    )


    context = SimpleNamespace(
        text="Example evidence."
    )


    messages = build_grounded_messages(
        query="Example question?",
        context=context,
    )


    user_prompt = messages[
        1
    ][
        "content"
    ]


    assert (
        "DIRECT_ANSWER: ...\n\nFACTS:"
        in
        user_prompt
    )


    assert (
        user_prompt.index(
            "DIRECT_ANSWER: ..."
        )
        <
        user_prompt.index(
            "FACTS:"
        )
    )