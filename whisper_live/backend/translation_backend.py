import json
import logging
import threading
import time
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
        self.model_lock = threading.Lock()
        self.latest_draft_segment = None
        self.last_draft_source_text = ""
        self.last_draft_sent_at = 0.0
        self.min_draft_interval_seconds = 0.6
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

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            model.eval()

            with self.model_lock:
                self.translation_model = model
            
            self.model_loaded = True
            logging.info(
                f"Translation model loaded successfully: {self.model_name}. "
                f"Target language: {self.target_language}"
            )
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            with self.model_lock:
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
            with self.model_lock:
                local_model = self.translation_model
                local_tokenizer = self.tokenizer

            if local_model is None or local_tokenizer is None:
                return text

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

            prompt = local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            encoded_input = local_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                generated_tokens = local_model.generate(
                    **encoded_input,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=local_tokenizer.pad_token_id,
                )

            prompt_len = encoded_input["input_ids"].shape[-1]
            new_tokens = generated_tokens[0][prompt_len:]
            output = local_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
                    
                # Translate all segments. Incomplete segments are sent as low-latency drafts.
                original_text = segment.get("text", "")
                is_completed = segment.get("completed", False)

                if not original_text.strip():
                    self.translation_queue.task_done()
                    continue

                if not is_completed:
                    now = time.time()
                    if original_text == self.last_draft_source_text:
                        self.translation_queue.task_done()
                        continue
                    if now - self.last_draft_sent_at < self.min_draft_interval_seconds:
                        self.translation_queue.task_done()
                        continue

                translated_text = self.translate_text(original_text)

                translated_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text,
                    "completed": is_completed,
                    "translation_state": "final" if is_completed else "draft",
                    "target_language": self.target_language
                }

                if is_completed:
                    self._upsert_translated_segment(translated_segment)
                    self.latest_draft_segment = None
                else:
                    self.latest_draft_segment = translated_segment
                    self.last_draft_source_text = original_text
                    self.last_draft_sent_at = now

                segments_to_send = self.prepare_translated_segments()
                self.send_translation_to_client(segments_to_send)
                
                self.translation_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error processing translation queue: {e}")
                continue
        
        logging.info(f"Translation processing ended for client {self.client_uid}")

    def _upsert_translated_segment(self, translated_segment):
        """Insert or replace a finalized translated segment by matching timestamps."""
        for i in range(len(self.translated_segments) - 1, -1, -1):
            seg = self.translated_segments[i]
            if seg["start"] == translated_segment["start"] and seg["end"] == translated_segment["end"]:
                self.translated_segments[i] = translated_segment
                return
        self.translated_segments.append(translated_segment)
    
    def prepare_translated_segments(self):
        """
        Prepare the last n translated segments to send to client.
        
        Returns:
            list: List of recent translated segments
        """
        if len(self.translated_segments) >= self.send_last_n_segments:
            segments = self.translated_segments[-self.send_last_n_segments:]
        else:
            segments = self.translated_segments[:]

        # Append current low-latency draft so the client can render it separately.
        if self.latest_draft_segment is not None:
            segments = segments + [self.latest_draft_segment]
        return segments
    
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
        self.latest_draft_segment = None
        self.last_draft_source_text = ""
        self.last_draft_sent_at = 0.0
        
        with self.model_lock:
            if self.translation_model:
                del self.translation_model
                self.translation_model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
        
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()
