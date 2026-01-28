# Standard prompt format keywords for position references
# These are used by token_positions.py for resolving position specs
PROMPT_KEYWORDS = {
    "situation": "SITUATION:",
    "task": "TASK:",
    "option_one": "OPTION_ONE:",
    "option_two": "OPTION_TWO:",
    "consider": "CONSIDER:",
    "action": "ACTION:",
    "format": "FORMAT:",
    "choice_prefix": "I select:",
    "reasoning_prefix": "My reasoning:",
}

# Keywords that should match LAST occurrence (in response, not FORMAT instructions)
LAST_OCCURRENCE_KEYWORDS = {"choice_prefix", "reasoning_prefix"}
