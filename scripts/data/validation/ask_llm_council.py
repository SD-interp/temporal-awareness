# Taken from https://huggingface.co/spaces/burtenshaw/karpathy-llm-council
import asyncio

def ask_llm_council(question: str, response_format=None) -> str:
    # FIXME: Not good decision, but works
    # 1. Load env from root dir
    from pathlib import Path
    from dotenv import load_dotenv
    root_dir = Path(__file__).parent / ".." / ".." / ".."
    env_file = str(root_dir / ".env.example")
    load_dotenv(env_file)

    # 2. Update Anthropic council model to Claude Opus 4.5:
    import llm_council.backend.config
    if "anthropic/claude-sonnet-4.5" in llm_council.backend.config.COUNCIL_MODELS:
        llm_council.backend.config.COUNCIL_MODELS.remove("anthropic/claude-sonnet-4.5")
    if "anthropic/claude-opus-4.5" not in llm_council.backend.config.COUNCIL_MODELS:
        llm_council.backend.config.COUNCIL_MODELS.append("anthropic/claude-opus-4.5")
    print(f"LLM council : {llm_council.backend.config.COUNCIL_MODELS}")
    print(f"Chairman model: {llm_council.backend.config.CHAIRMAN_MODEL}")

    # 3. Import llm_council backend
    from llm_council.backend.council import run_full_council
    from llm_council.backend.config import COUNCIL_MODELS, CHAIRMAN_MODEL

    """
    Ask the LLM Council a question.

    The council consists of multiple advanced LLMs (currently: {models}) that:
    1. Individually answer the question
    2. Rank each other's answers
    3. Synthesize a final best answer (Chairman: {chairman})

    Args:
        question: The user's question to be discussed by the council.

    Returns:
        The final synthesized answer from the Council Chairman.
    """.format(models=", ".join([m.split("/")[-1] for m in COUNCIL_MODELS]), chairman=CHAIRMAN_MODEL.split("/")[-1])

    try:
        # Run the council
        # run_full_council returns (stage1, stage2, stage3, metadata)
        _, _, stage3_result, _ = asyncio.run(run_full_council(question, response_format))

        response = stage3_result.get("response")
        if not response:
            return "The council failed to generate a response."

        return response

    except Exception as e:
        return f"Error consulting the council: {str(e)}"

def get_council():
    import llm_council.backend.config
    if "anthropic/claude-sonnet-4.5" in llm_council.backend.config.COUNCIL_MODELS:
        llm_council.backend.config.COUNCIL_MODELS.remove("anthropic/claude-sonnet-4.5")
    if "anthropic/claude-opus-4.5" not in llm_council.backend.config.COUNCIL_MODELS:
        llm_council.backend.config.COUNCIL_MODELS.append("anthropic/claude-opus-4.5")
    return {"LLM council": llm_council.backend.config.COUNCIL_MODELS,
            "Chairman": llm_council.backend.config.CHAIRMAN_MODEL}
