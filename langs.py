"""Language registry: Whisper (ISO-639-1) <-> NLLB-200 (FLORES-200) codes."""

# iso: (flores_code, display name)
LANGUAGES = {
    "ru": ("rus_Cyrl", "Русский"),
    "en": ("eng_Latn", "English"),
    "cs": ("ces_Latn", "Čeština"),
    "uk": ("ukr_Cyrl", "Українська"),
    "pl": ("pol_Latn", "Polski"),
    "sk": ("slk_Latn", "Slovenčina"),
    "de": ("deu_Latn", "Deutsch"),
    "fr": ("fra_Latn", "Français"),
    "es": ("spa_Latn", "Español"),
    "it": ("ita_Latn", "Italiano"),
    "pt": ("por_Latn", "Português"),
    "nl": ("nld_Latn", "Nederlands"),
    "sv": ("swe_Latn", "Svenska"),
    "da": ("dan_Latn", "Dansk"),
    "no": ("nob_Latn", "Norsk"),
    "fi": ("fin_Latn", "Suomi"),
    "hu": ("hun_Latn", "Magyar"),
    "ro": ("ron_Latn", "Română"),
    "bg": ("bul_Cyrl", "Български"),
    "sr": ("srp_Cyrl", "Српски"),
    "hr": ("hrv_Latn", "Hrvatski"),
    "sl": ("slv_Latn", "Slovenščina"),
    "el": ("ell_Grek", "Ελληνικά"),
    "tr": ("tur_Latn", "Türkçe"),
    "he": ("heb_Hebr", "עברית"),
    "ar": ("arb_Arab", "العربية"),
    "fa": ("pes_Arab", "فارسی"),
    "hi": ("hin_Deva", "हिन्दी"),
    "ja": ("jpn_Jpan", "日本語"),
    "ko": ("kor_Hang", "한국어"),
    "zh": ("zho_Hans", "中文"),
    "vi": ("vie_Latn", "Tiếng Việt"),
    "th": ("tha_Thai", "ไทย"),
    "id": ("ind_Latn", "Indonesia"),
    "be": ("bel_Cyrl", "Беларуская"),
    "kk": ("kaz_Cyrl", "Қазақша"),
    "lt": ("lit_Latn", "Lietuvių"),
    "lv": ("lvs_Latn", "Latviešu"),
    "et": ("est_Latn", "Eesti"),
    "ca": ("cat_Latn", "Català"),
}


def flores(iso: str) -> str | None:
    """Whisper language code -> NLLB code. None if NLLB has no such language."""
    entry = LANGUAGES.get(iso)
    return entry[0] if entry else None


def display(iso: str) -> str:
    entry = LANGUAGES.get(iso)
    return entry[1] if entry else iso


def picker_items() -> list[tuple[str, str]]:
    """(iso, label) pairs for the target-language dropdown, sorted by label."""
    return sorted(
        ((iso, name) for iso, (_, name) in LANGUAGES.items()),
        key=lambda pair: pair[1].lower(),
    )
