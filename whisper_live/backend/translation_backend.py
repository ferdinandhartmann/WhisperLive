import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from whisper_live.backend.base import ServeClientBase


class ServeClientTranslation(ServeClientBase):
    """
    Handles translation of completed transcription segments in a separate thread.
    Reads from a queue populated by the transcription backend and sends translated
    segments back to the client via WebSocket.
    """
    
    def __init__(
        self,
        client_uid,
        websocket,
        translation_queue,
        target_language="en",
        send_last_n_segments=15,
        model_name="Qwen/Qwen2.5-14B-Instruct",
        translation_system_prompt=None,
        initial_prompt=None,
    ):
        """
        Initialize the translation client.
        
        Args:
            client_uid (str): Unique identifier for the client
            websocket: WebSocket connection to the client
            translation_queue (queue.Queue): Queue containing completed segments to translate
            target_language (str): Target language code (default: "en" for English)
            send_last_n_segments (int): Number of recent translated segments to send
            model_name (str): Translation model name to use
        """
        super().__init__(client_uid, websocket, send_last_n_segments)
        self.translation_queue = translation_queue
        self.target_language = target_language
        self.model_name = model_name
        self.translation_system_prompt = (
            translation_system_prompt
            or "You are an expert Japanese-to-English translator. "
               "Translate the user's Japanese input into natural, conversational English. "
               "Preserve the original tone, intent, and nuance. "
               "Do not produce a literal word-for-word translation. "
               "Return only the final English translation with no extra commentary."
        )
        self.initial_prompt = initial_prompt
        self.translated_segments = []
        self.translation_model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.load_translation_model()
        
    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Loading translation model on : {self.device}")

            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.translation_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.translation_model.eval()
            
            self.model_loaded = True
            logging.info(
                f"Translation model loaded successfully: {self.model_name}. "
                f"Target language: {self.target_language}"
            )
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            self.translation_model = None
            self.tokenizer = None
            self.model_loaded = False
    
    def translate_text(self, text: str) -> str:
        """
        Translate a single text segment.
        
        Args:
            text (str): Text to translate
            
        Returns:
            str: Translated text or original text if translation fails
        """
        if not self.model_loaded or not text.strip():
            return text
            
        try:
            text = self.clean_text(text)

            messages = [
                {"role": "system", "content": self.translation_system_prompt},
                {
                    "role": "system",
                    "content": (
                        f"Translate from Japanese to {self.target_language}. "
                        "Keep meaning and tone, and return only the translation."
                    ),
                },
            ]
            if self.initial_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Additional context: {self.initial_prompt}",
                    }
                )
            messages.append({"role": "user", "content": text})

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            encoded_input = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                generated_tokens = self.translation_model.generate(
                    **encoded_input,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            prompt_len = encoded_input["input_ids"].shape[-1]
            new_tokens = generated_tokens[0][prompt_len:]
            output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return output if output else text
            
        except Exception as e:
            logging.error(f"Translation failed for text '{text}': {e}")
            return text
        
    def clean_text(self, text):
        FILLERS = ["なんか", "じゃん", "かな", "えっと"]
        for f in FILLERS:
            text = text.replace(f, "")
        return text
    
    def process_translation_queue(self):
        """
        Process segments from the translation queue.
        Continuously reads from the queue until None is received (exit signal).
        """
        logging.info(f"Starting translation processing for client {self.client_uid}")
        
        while not self.exit:
            try:
                # Blocking get keeps latency low and avoids polling delays.
                segment = self.translation_queue.get()
                
                # Check for exit signal
                if segment is None:
                    logging.info(f"Received exit signal for translation client {self.client_uid}")
                    break
                    
                # Only translate completed segments
                if not segment.get("completed", False):
                    self.translation_queue.task_done()
                    continue
                    
                # Translate the segment
                original_text = segment.get("text", "")
                translated_text = self.translate_text(original_text)
                
                # Create translated segment
                translated_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text,
                    "completed": segment.get("completed", False),
                    "target_language": self.target_language
                }
                
                self.translated_segments.append(translated_segment)
                segments_to_send = self.prepare_translated_segments()
                self.send_translation_to_client(segments_to_send)
                
                self.translation_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error processing translation queue: {e}")
                continue
        
        logging.info(f"Translation processing ended for client {self.client_uid}")
    
    def prepare_translated_segments(self):
        """
        Prepare the last n translated segments to send to client.
        
        Returns:
            list: List of recent translated segments
        """
        if len(self.translated_segments) >= self.send_last_n_segments:
            return self.translated_segments[-self.send_last_n_segments:]
        return self.translated_segments[:]
    
    def send_translation_to_client(self, translated_segments):
        """
        Send translated segments to the client via WebSocket.
        
        Args:
            translated_segments (list): List of translated segments to send
        """
        self._safe_send(
            json.dumps({
                "uid": self.client_uid,
                "translated_segments": translated_segments,
            }),
            log_prefix="Sending translation data to client"
        )
    
    def speech_to_text(self):
        """
        Override parent method to handle translation processing.
        This method will be called when the translation thread starts.
        """
        self.process_translation_queue()
    
    def set_target_language(self, language: str):
        """
        Change the target language for translation.
        
        Args:
            language (str): New target language code
        """
        self.target_language = language
        logging.info(f"Target language changed to: {language}")
    
    def cleanup(self):
        """Clean up translation resources."""
        logging.info(f"Cleaning up translation resources for client {self.client_uid}")
        self.exit = True
        
        try:
            self.translation_queue.put(None, timeout=1.0)
        except Exception:
            pass
        
        self.translated_segments.clear()
        
        if self.translation_model:
            del self.translation_model
            self.translation_model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()
