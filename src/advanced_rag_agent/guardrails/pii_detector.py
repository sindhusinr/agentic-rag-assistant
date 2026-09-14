from presidio_analyzer import AnalyzerEngine

# Load Presidio once and reuse it
analyzer = AnalyzerEngine()

PII_TYPES = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
]

MIN_SCORE = 0.6


def detect_pii(text: str):
    # Detect only configured PII types
    results = analyzer.analyze(
        text=text,
        entities=PII_TYPES,
        language="en",
    )

    return [
        result
        for result in results
        if result.score >= MIN_SCORE
    ]


def mask_pii(text: str, results) -> str:
    # Replace from right to left so indexes remain valid
    masked_text = text

    for result in sorted(
        results,
        key=lambda x: x.start,
        reverse=True,
    ):
        placeholder = f"<{result.entity_type}>"

        masked_text = (
            masked_text[:result.start]
            + placeholder
            + masked_text[result.end:]
        )

    return masked_text