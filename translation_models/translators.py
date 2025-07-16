import os
import time
from collections import deque
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
import json

try:
    import google.generativeai as genai
except ImportError:
    genai = None

class GeminiTranslator:
    # Class-level rate limiters (shared across all instances)
    _request_times = deque()
    _token_times = deque()
    _day_request_times = deque()
    _RPM = 30
    _TPM = 1_000_000
    _RPD = 200
    _window_seconds = 60
    _day_seconds = 24 * 60 * 60

    def __init__(self, target_language, api_key=None):
        if genai is None:
            raise ImportError("google-generativeai is not installed. Please install it via pip.")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not provided as an argument or set as an environment variable.")
        # Configure API key if possible
        configure_fn = getattr(genai, 'configure', None)
        if callable(configure_fn):
            configure_fn(api_key=self.api_key)
        else:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        self.target_language = target_language
        self.model_name = "gemini-2.0-flash-lite"

    @classmethod
    def _rate_limit(cls, tokens=0):
        now = time.time()
        # Clean up old entries
        while cls._request_times and now - cls._request_times[0] > cls._window_seconds:
            cls._request_times.popleft()
        while cls._token_times and now - cls._token_times[0][0] > cls._window_seconds:
            cls._token_times.popleft()
        while cls._day_request_times and now - cls._day_request_times[0] > cls._day_seconds:
            cls._day_request_times.popleft()
        # Check limits
        if len(cls._request_times) >= cls._RPM:
            sleep_time = cls._window_seconds - (now - cls._request_times[0]) + 0.1
            time.sleep(max(sleep_time, 0))
            return cls._rate_limit(tokens)
        if sum(t for _, t in cls._token_times) + tokens > cls._TPM:
            sleep_time = cls._window_seconds - (now - cls._token_times[0][0]) + 0.1
            time.sleep(max(sleep_time, 0))
            return cls._rate_limit(tokens)
        if len(cls._day_request_times) >= cls._RPD:
            raise RuntimeError("Gemini API daily request limit reached (200 RPD). Try again tomorrow.")
        # Record this request
        cls._request_times.append(now)
        cls._token_times.append((now, tokens))
        cls._day_request_times.append(now)

    def _count_tokens(self, text):
        # Simple token estimate: 1 token ≈ 4 chars (for English)
        return max(1, len(text) // 4)

    def translate(self, text):
        if not text.strip():
            return ""
        tokens = self._count_tokens(text)
        self._rate_limit(tokens)
        prompt = f"Translate the following English text to {self.target_language}. Only output the translation, do not add any extra text. If the input is empty, output should be empty. Text: {text}"
        GenerativeModel = getattr(genai, 'GenerativeModel', None)
        if GenerativeModel is None:
            raise ImportError("Your google-generativeai version does not support GenerativeModel. Please upgrade the package.")
        model = GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, 'text') else str(response).strip()

    def batch_translate(self, texts):
        # Remove empty/whitespace-only texts, but keep their positions for output
        indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in indices]
        if not non_empty_texts:
            return ["" for _ in texts]
        # Estimate tokens for all texts
        total_tokens = sum(self._count_tokens(t) for t in non_empty_texts)
        self._rate_limit(total_tokens)
        # Prepare a single prompt for all texts
        prompt = (
            f"Translate the following English texts to {self.target_language}. "
            "Return only the translations, in the same order, separated by <SEP>. "
            "If an input is empty, output should be empty.\n"
        )
        for i, t in enumerate(non_empty_texts):
            prompt += f"Text {i+1}: {t}\n"
        prompt += "Output:"
        GenerativeModel = getattr(genai, 'GenerativeModel', None)
        if GenerativeModel is None:
            raise ImportError("Your google-generativeai version does not support GenerativeModel. Please upgrade the package.")
        model = GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        output = response.text.strip() if hasattr(response, 'text') else str(response).strip()
        # Split by <SEP> and map back to original positions
        translations = output.split('<SEP>')
        translations = [t.strip() for t in translations]
        # Fill in empty translations for empty/whitespace-only inputs
        result = ["" for _ in texts]
        for idx, orig_idx in enumerate(indices):
            result[orig_idx] = translations[idx] if idx < len(translations) else ""
        return result

class EnglishToSpanishTranslator:
    def __init__(self):
        model_name = "Helsinki-NLP/opus-mt-en-es"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
    def translate(self, text):
        if not text.strip():
            return ""
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class QwenJapaneseTranslator:
    # Class-level rate limiters (shared across all instances)
    _minute_requests = []  # list of timestamps
    _day_requests = []     # list of timestamps
    _MAX_PER_MIN = 20
    _MAX_PER_DAY = 50
    _MINUTE = 60
    _DAY = 24 * 60 * 60

    def __init__(self, api_key, site_url=None, site_title=None):
        self.api_key = api_key
        self.site_url = site_url
        self.site_title = site_title
        self.model = "qwen/qwen3-32b:free"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _rate_limit(self):
        now = time.time()
        # Clean up old entries
        QwenJapaneseTranslator._minute_requests = [t for t in QwenJapaneseTranslator._minute_requests if now - t < self._MINUTE]
        QwenJapaneseTranslator._day_requests = [t for t in QwenJapaneseTranslator._day_requests if now - t < self._DAY]
        # Check per-minute limit
        if len(QwenJapaneseTranslator._minute_requests) >= self._MAX_PER_MIN:
            wait_time = self._MINUTE - (now - QwenJapaneseTranslator._minute_requests[0]) + 0.1
            print(f"[Qwen Rate Limit] Hit 20 requests/min. Delaying for {wait_time:.1f} seconds.")
            time.sleep(max(wait_time, 0))
            return self._rate_limit()
        # Check per-day limit
        if len(QwenJapaneseTranslator._day_requests) >= self._MAX_PER_DAY:
            wait_time = self._DAY - (now - QwenJapaneseTranslator._day_requests[0]) + 0.1
            print(f"[Qwen Rate Limit] Hit 50 requests/day. Delaying for {wait_time/3600:.2f} hours.")
            time.sleep(max(wait_time, 0))
            return self._rate_limit()
        # Record this request
        QwenJapaneseTranslator._minute_requests.append(now)
        QwenJapaneseTranslator._day_requests.append(now)

    def translate(self, text):
        if not text.strip():
            return ""
        self._rate_limit()
        prompt = f"Translate the following English text to Japanese. Only output the translation, do not add any extra text. If the input is empty, output should be empty. Text: {text}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_title:
            headers["X-Title"] = self.site_title
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            # The response format may vary; adjust as needed
            try:
                return result["choices"][0]["message"]["content"].strip()
            except Exception:
                return "[Qwen API: Unexpected response]"
        else:
            return f"[Qwen API Error: {response.status_code}]"

class GoogleTranslateTranslator:
    def __init__(self, target_lang):
        self.target_lang = target_lang
        self.api_url = "https://translate.googleapis.com/translate_a/single"

    def translate(self, text):
        if not text.strip():
            return ""
        params = {
            'client': 'gtx',
            'sl': 'en',
            'tl': self.target_lang,
            'dt': 't',
            'q': text
        }
        response = requests.get(self.api_url, params=params)
        if response.status_code == 200:
            try:
                result = response.json()
                return ''.join([item[0] for item in result[0] if item[0]])
            except Exception:
                return "[Google Translate: Unexpected response]"
        else:
            return f"[Google Translate Error: {response.status_code}]"