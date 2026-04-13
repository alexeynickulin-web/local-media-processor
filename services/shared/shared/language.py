"""Language codes and validation utilities."""

from .exceptions import ValidationError


# ISO 639-1 language codes supported by the system
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "hi": "Hindi",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
}

# NLLB language codes (may differ from ISO 639-1)
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "hi": "hin_Deva",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
    "ms": "msa_Latn",
    "sv": "swe_Latn",
    "no": "nob_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
}


def validate_language_code(code: str) -> str:
    """Validate and normalize a language code.

    Args:
        code: Language code (e.g., "en", "ru")

    Returns:
        Normalized language code

    Raises:
        ValidationError: If language code not supported
    """
    if not code:
        raise ValidationError("Language code cannot be empty")

    code = code.lower().strip()

    if code not in SUPPORTED_LANGUAGES:
        raise ValidationError(
            f"Unsupported language code: '{code}'. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
        )

    return code


def get_nllb_code(code: str) -> str:
    """Convert ISO 639-1 code to NLLB language code.

    Args:
        code: ISO 639-1 language code

    Returns:
        NLLB language code

    Raises:
        ValidationError: If language code not mapped
    """
    code = validate_language_code(code)
    nllb_code = NLLB_LANGUAGE_CODES.get(code)

    if not nllb_code:
        raise ValidationError(f"No NLLB mapping for language: '{code}'")

    return nllb_code


def get_language_name(code: str) -> str:
    """Get human-readable language name.

    Args:
        code: Language code

    Returns:
        Language name
    """
    code = validate_language_code(code)
    return SUPPORTED_LANGUAGES.get(code, code)
