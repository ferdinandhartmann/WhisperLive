from whisper_live.client import TranscriptionClient

def main():
    client = TranscriptionClient(
        host="localhost",
        port=9090,
        
        use_vad=True,
        
        model="large-v3", #large-v3-turbo not multilingual
        lang="ja",                
        translate=False,    
                
        # output_recording_filename="./rec.wav",
        save_output_recording=True,
        
        enable_translation=True,
        target_language="en",
        translation_model_name="Qwen/Qwen2.5-7B-Instruct", #Qwen/Qwen2.5-14B-Instruct
        translation_system_prompt=(
            "You are an expert Japanese-to-English translator. "
            "Translate Japanese speech into natural, conversational English. "
            "Preserve tone and meaning, avoid literal phrasing, and output only the final translation."
        ),

        # It is translating the segments added sinc the last translation. Segment = full sentence.
        enable_deepl_translation=False,
        deepl_source_language="JA",
        deepl_target_language="EN-US",
        deepl_translation_interval=2, # seconds
        # deepl_translation_srt_file_path="./output_deepl_translated.srt",

        enable_gemini_translation=False,
        gemini_target_language="en",
        gemini_translation_interval=15, # seconds
        gemini_model="gemini-1.5-flash",
        # gemini_translation_output_path="./output_gemini_translation.txt",
        
        initial_prompt=(
            "This is spoken Japanese conversation. "
            "For Japanese transcription, restore natural sentence punctuation (。 and ？) at sentence boundaries."
        ),
        # initial_prompt="Translate german sentences into english.",
        # initial_prompt="Translate this japanese lab meeting conversation about robotics and AI models into english.",
    )

    client()  # starts microphone capture
    
    # file_path = r"C:\Users\ferdi\Documents\programming\WhisperLive\podcast_german.mp3"
    # # 3. Pass the file path to the client call
    # print(f"Sending file to server: {file_path}")
    # client(file_path)

if __name__ == "__main__":
    main()
