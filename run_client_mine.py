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
        
        enable_translation=False,
        target_language="en",

        enable_deepl_translation=True,
        deepl_source_language="JA",
        deepl_target_language="EN-US",
        deepl_translation_interval=10, # seconds
        # deepl_translation_srt_file_path="./output_deepl_translated.srt",

        enable_gemini_translation=False,
        gemini_target_language="en",
        gemini_translation_interval=15, # seconds
        gemini_model="gemini-1.5-flash",
        # gemini_translation_output_path="./output_gemini_translation.txt",
        
        initial_prompt=None,
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
