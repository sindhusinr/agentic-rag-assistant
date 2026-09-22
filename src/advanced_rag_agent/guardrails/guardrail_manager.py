from advanced_rag_agent.guardrails.prompt_injection import detect_prompt_injection
from advanced_rag_agent.guardrails.pii_detector import detect_pii,mask_pii


def validate_input(text: str) -> dict:
    """
    Validate and sanitize user input before processing.
    """

    # Block prompt-injection attempts before any LLM call
    if detect_prompt_injection(text):
        return {
            "allowed": False,
            "reason": "Prompt injection detected.",
            "sanitized_text": None,
            "pii_entities": [],
        }

    # Mask PII instead of blocking a valid request
    pii_results = detect_pii(text)

    pii_entities = list({
        result.entity_type
        for result in pii_results
    })

    sanitized_text = mask_pii(
        text=text,
        results=pii_results,
    )

    return {
        "allowed": True,
        "reason": None,
        "sanitized_text": sanitized_text,
        "pii_entities": pii_entities,
    }


def validate_output(text: str) -> dict:
    """
    Sanitize generated output before returning it to the user.
    """

    pii_results = detect_pii(text)

    pii_entities = list({
        result.entity_type
        for result in pii_results
    })

    # Prevent accidental PII leakage in generated responses
    sanitized_text = mask_pii(
        text=text,
        results=pii_results,
    )

    return {
        "sanitized_text": sanitized_text,
        "pii_entities": pii_entities,
    }