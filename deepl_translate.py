import os
import time
import deepl

SOURCE_LANG = "EN"
TARGET_LANG = "DE"

print("Translating with DeepL (batched)...")

deepl_translator = deepl.Translator(os.environ["DEEPL_API_KEY"])

deepl_translations = []

BATCH_SIZE = 40  # safe size



results = deepl_translator.translate_text(
    text,
    source_lang=SOURCE_LANG,
    target_lang=TARGET_LANG
)

deepl_translations.extend([r.text for r in results])

print(f"  Translated {len(deepl_translations)}/{len(sentences)} sentences")
time.sleep(0.1)  # polite throttling
