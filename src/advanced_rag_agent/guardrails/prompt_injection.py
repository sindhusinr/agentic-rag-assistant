import re

INJECTION_PATTERNS = [
    r"ignore\s+(the\s+)?(all\s+|previous\s+)?instructions?",
    r"forget\s+(the\s+)?(all\s+|previous\s+)?instructions?",
    r"disregard\s+(the\s+)?(all\s+|previous\s+)?instructions?",
    r"override\s+(the\s+)?(all\s+|previous\s+)?instructions?",
    r"act\s+as\s+(the\s+)?system",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?system\s+prompt",
    r"developer\s+message",
    r"hidden\s+instructions?",
    r"jailbreak",
    r"bypass\s+(the\s+)?security",
    r"disable\s+(the\s+)?safeguards?",
]


def detect_prompt_injection(query: str) -> bool:
    # Fast deterministic check before any LLM call
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return True

    return False