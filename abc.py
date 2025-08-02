from deep_translator import GoogleTranslator

# List of languages to test
languages = {
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Bengali": "bn"
}

# Text to translate
text_to_translate = "Hello i am good, how are you"

# Loop through each language and test translation
for language_name, language_code in languages.items():
    try:
        # Perform translation
        translated_text = GoogleTranslator(source="en", target=language_code).translate(text_to_translate)
        print(f"{language_name} Translation: {translated_text}")
    except Exception as e:
        print(f"Translation failed for {language_name}: {e}")
