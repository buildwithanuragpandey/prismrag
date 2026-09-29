from adaptive_agentic_rag.orchestration.state import (
    AgentState,
    create_initial_state
)


def main():

    state: AgentState = create_initial_state(
        query=(
            "Compare Amazon and Walmart deals"
        ),
        max_retries=1
    )


    print(
        "\n===== INITIAL STATE ====="
    )


    for key, value in state.items():

        print(
            f"{key}: {value}"
        )


    assert (
        state["original_query"]
        ==
        "Compare Amazon and Walmart deals"
    )


    assert (
        state["current_query"]
        ==
        state["original_query"]
    )


    assert (
        state["retry_count"]
        ==
        0
    )


    assert (
        state["max_retries"]
        ==
        1
    )


    assert (
        state["retrieved_results"]
        ==
        []
    )


    assert (
        state["evidence_sufficient"]
        is None
    )


    assert (
        state["final_answer"]
        is None
    )


    print(
        "\nState initialization: OK"
    )


if __name__ == "__main__":

    main()