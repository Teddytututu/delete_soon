import os
import requests
import openai
from dotenv import load_dotenv
import json
import threading
import random  # P0-1 FIX: For jitter in rate limit backoff
import time  # P0-2 FIX: For rate limiting
from collections import deque
from typing import Optional, Dict, List, Any

load_dotenv()

# P0-2 FIX: Global rate limiting to prevent Error 429
# Tracks active API calls across all LLM instances
_global_api_lock = threading.Lock()
# P0 FIX: Completely serialize ALL API calls globally to prevent 429 errors
# Multiple LLM instances across pipeline stages must not call API concurrently
# Each API call acquires the global lock, holds it during the API call + 1s delay, then releases


class LLM:

    def __init__(self, model_name, key, logger=None, user_id=None,
                 logger_manager=None, tracker=None, max_usage_history: int = 1000,
                 enable_latent_summary=False, latent_summary_stages=None,
                 enable_summary_cache=True, cache_similarity_threshold=0.85):
        """
        Initialize LLM client with comprehensive tracking and rate limiting.

        Args:
            model_name: Model identifier (e.g., 'gpt-4o', 'glm-4.7')
            key: API key for the model
            logger: Deprecated - use logger_manager instead
            user_id: Optional user identifier
            logger_manager: Optional MMExperimentLogger for structured logging
            tracker: Optional ExecutionTracker for detailed event tracking
            max_usage_history: Maximum number of usage records to keep (LRU eviction)
            enable_latent_summary: Whether to generate LLM-based summaries for each API call
            latent_summary_stages: List of stages to generate summaries for (None = all stages)
                                  Example: ['Problem Analysis', 'Mathematical Modeling']
            enable_summary_cache: Whether to cache and reuse similar summaries
            cache_similarity_threshold: Similarity threshold for cache reuse (0.0-1.0)
        """
        self.model_name = model_name
        self.logger = logger  # Deprecated but kept for backward compatibility
        self.logger_manager = logger_manager  # New structured logging
        self.tracker = tracker  # Execution tracking
        self.user_id = user_id
        self.api_key = key
        self.enable_latent_summary = enable_latent_summary  # Enable latent summaries
        self.latent_summary_stages = latent_summary_stages  # Filter which stages get summaries
        self.enable_summary_cache = enable_summary_cache  # Enable summary caching
        self.cache_similarity_threshold = cache_similarity_threshold  # Cache threshold

        # CRITICAL: Initialize instance-level usage tracking with LRU eviction
        self._max_usage_history = max_usage_history
        self._usages = deque(maxlen=max_usage_history)

        # CRITICAL: Initialize threading lock to serialize API calls (prevent Error 429)
        self.api_lock = threading.Lock()

        # Initialize summary cache
        self._summary_cache = {}  # {prompt_hash: summary}

        # Set API base URL based on model name
        if self.model_name in ['deepseek-chat', 'deepseek-reasoner']:
            self.api_base = os.getenv('DEEPSEEK_API_BASE')
        elif self.model_name in ['gpt-4o', 'gpt-4']:
            self.api_base = os.getenv('OPENAI_API_BASE')
        elif self.model_name in ['glm-4-plus', 'glm-4-0520', 'glm-4-air', 'glm-4-flash', 'glm-4.7']:
            # NEW: GLM-4.7 model support
            self.api_base = "https://open.bigmodel.cn/api/paas/v4"
        elif self.model_name in ['qwen2.5-72b-instruct']:
            self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            # Default to OpenAI if model not recognized
            self.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

        # Validate API key
        if not self.api_key:
            raise ValueError('API key not found')

        # Create OpenAI client (works with OpenAI-compatible APIs)
        self.client = openai.Client(api_key=self.api_key, base_url=self.api_base)

    @property
    def usages(self) -> List[Dict[str, Any]]:
        """
        Backward compatibility property for accessing usage history.

        Returns a copy of the usage deque as a list.
        Deprecated: Use get_total_usage() method instead.
        """
        return list(self._usages)

    @usages.setter
    def usages(self, value):
        """
        Backward compatibility setter for usage history.

        Converts list/deque to bounded deque to prevent memory leaks.
        Discouraged: Use internal methods instead.

        Raises:
            TypeError: If value is not a list or deque
        """
        if isinstance(value, deque):
            self._usages = value
        elif isinstance(value, list):
            # Convert to deque with LRU eviction
            self._usages = deque(value, maxlen=self._max_usage_history)
        else:
            raise TypeError(f"usages must be a list or deque, got {type(value)}")

    def reset(self, api_key=None, api_base=None, model_name=None):
        if api_key:
            self.api_key = api_key
        if api_base:
            self.api_base = api_base
        if model_name:
            self.model_name = model_name
        self.client = openai.Client(api_key=self.api_key, base_url=self.api_base)

    def generate(self, prompt, system='', usage=True, temperature=None):
        """
        Generate response from LLM with rate limiting, retry logic, and tracking.

        Args:
            prompt: User prompt
            system: Optional system message
            usage: Whether to track usage statistics
            temperature: Optional temperature for randomness control (0.0=deterministic, 1.0=creative)
                        If None, uses default 0.7. For reproducible results, use temperature=0.0

        Returns:
            Generated response text

        P0 FIX #1: Added exponential backoff retry for RateLimitError 429
        """
        # Use default temperature if not specified
        if temperature is None:
            temperature = 0.7

        # P0 FIX: Use global lock to completely serialize API calls
        global _global_api_lock

        # ======== P0 FIX: Exponential Backoff Retry for 429 Errors ========
        max_retries = 3
        base_delay = 2.0  # Initial delay 2 seconds

        for attempt in range(max_retries):
            try:
                # P0 FIX: Acquire global lock to COMPLETELY serialize all API calls
                # This prevents ANY concurrent API calls across all LLM instances
                with _global_api_lock:
                    # CRITICAL: Add 1-second delay to prevent rate limiting
                    time.sleep(1)

                    # Track LLM call start (if tracker available)
                    start_time = None
                    if self.tracker:
                        start_time = time.time()

                    # API call (now serialized and rate-limited)
                    if self.model_name in ['deepseek-chat', 'deepseek-reasoner']:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=temperature,  # Use parameter instead of hardcoded 0.7
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )
                        answer = response.choices[0].message.content
                        usage_data = {
                            'completion_tokens': response.usage.completion_tokens,
                            'prompt_tokens': response.usage.prompt_tokens,
                            'total_tokens': response.usage.total_tokens
                        }
                    elif self.model_name in ['gpt-4o', 'gpt-4']:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=temperature,  # Use parameter instead of hardcoded 0.7
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )
                        answer = response.choices[0].message.content
                        usage_data = {
                            'completion_tokens': response.usage.completion_tokens,
                            'prompt_tokens': response.usage.prompt_tokens,
                            'total_tokens': response.usage.total_tokens
                        }
                    elif self.model_name in ['glm-4-plus', 'glm-4-0520', 'glm-4-air', 'glm-4-flash', 'glm-4.7']:
                        # NEW: GLM-4.7 support
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=temperature,  # Use parameter instead of hardcoded 0.7
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )
                        answer = response.choices[0].message.content
                        usage_data = {
                            'completion_tokens': response.usage.completion_tokens,
                            'prompt_tokens': response.usage.prompt_tokens,
                            'total_tokens': response.usage.total_tokens
                        }
                    elif self.model_name in ['qwen2.5-72b-instruct']:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=temperature,  # Use parameter instead of hardcoded 0.7
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0
                        )
                        answer = response.choices[0].message.content
                        usage_data = {
                            'completion_tokens': response.usage.completion_tokens,
                            'prompt_tokens': response.usage.prompt_tokens,
                            'total_tokens': response.usage.total_tokens
                        }
                    else:
                        # Default fallback
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {'role': 'system', 'content': system},
                                {'role': 'user', 'content': prompt}
                            ],
                            temperature=0.7
                        )
                        answer = response.choices[0].message.content
                        usage_data = {
                            'completion_tokens': response.usage.completion_tokens,
                            'prompt_tokens': response.usage.prompt_tokens,
                            'total_tokens': response.usage.total_tokens
                        }

                    # Track usage (if enabled)
                    usage_data_with_meta = {
                        **usage_data,
                        'model': self.model_name,
                        'timestamp': time.time()
                    }
                    if usage:
                        self._usages.append(usage_data_with_meta)

                    # Generate latent summary if enabled (before tracking)
                    latent_summary = None
                    if self.enable_latent_summary:
                        latent_summary = self._generate_latent_summary(prompt, answer, usage_data)

                    # Track LLM call completion (if tracker available)
                    if self.tracker and start_time:
                        latency = time.time() - start_time
                        self.tracker.track_llm_call(
                            prompt=prompt,
                            response=answer,
                            model=self.model_name,
                            usage=usage_data_with_meta,
                            stage=getattr(self, '_current_stage', 'unknown'),
                            latency=latency,
                            latent_summary=latent_summary  # NEW: Pass latent summary to tracker
                        )

                    # Log LLM call (if logger available)
                    if self.logger_manager:
                        self.logger_manager.get_logger('llm_api').info(
                            f"LLM call: model={self.model_name}, "
                            f"prompt_tokens={usage_data['prompt_tokens']}, "
                            f"completion_tokens={usage_data['completion_tokens']}"
                        )

                    elif self.logger:
                        self.logger.info(f"[LLM] UserID: {self.user_id}, Model: {self.model_name}, Usage: {usage_data}")

                    return answer  # Success - return immediately

            except Exception as e:
                error_str = str(e)
                error_code = getattr(e, 'code', None)

                # P0 FIX #1: Check if this is a 429 rate limit error
                is_rate_limit = ('429' in error_str or '1302' in error_str or
                                error_code in [429, 1302] or
                                'rate_limit' in error_str.lower() or
                                'RateLimitError' in type(e).__name__)

                if is_rate_limit:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s + random jitter
                        delay = base_delay * (2 ** attempt)
                        jitter = random.random()  # P0-1 FIX: Was time.random(), now random.random()
                        total_delay = delay + jitter

                        if self.logger_manager:
                            self.logger_manager.get_logger('llm_api').warning(
                                f"[P0 FIX #1] Rate limit hit (429/1302). "
                                f"Retrying in {total_delay:.1f}s (attempt {attempt+1}/{max_retries})"
                            )
                        elif self.logger:
                            self.logger.warning(
                                f"[P0 FIX #1] Rate limit hit. Retrying in {total_delay:.1f}s "
                                f"(attempt {attempt+1}/{max_retries})"
                            )

                        time.sleep(total_delay)
                        continue  # Retry the API call

                # If not a rate limit error, or retry attempts exhausted, re-raise the exception
                raise
        # ==========================================

    def get_total_usage(self):
        """
        Calculate total tokens used across all API calls.

        Returns:
            Total token count
        """
        total_usage = {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'total_tokens': 0
        }
        # Use instance-level _usages instead of global usages
        for usage in self._usages:
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
        return total_usage

    def clear_usage(self):
        """
        Clear usage history while maintaining deque instance.
        """
        self._usages.clear()  # Clear deque without reassigning

    def _generate_latent_summary(self, prompt: str, response: str, usage_data: dict):
        """
        Generate LLM-based latent summary for each API call.

        This creates concise summaries of what each LLM call accomplished,
        making it easier to understand the execution flow.

        Args:
            prompt: The prompt sent to LLM
            response: The LLM's response
            usage_data: Token usage data

        Returns:
            Summary string, or None if generation failed or filtered out
        """
        import time
        from pathlib import Path
        import hashlib

        try:
            # Get current stage and task
            stage = getattr(self, '_current_stage', 'unknown')
            task = getattr(self, '_current_task', 'unknown')

            # FEATURE 1: Selective summarization - check if stage is in filter list
            if self.latent_summary_stages is not None:
                if stage not in self.latent_summary_stages:
                    # Stage not in filter list, skip summarization
                    return None

            # FEATURE 2: Summary caching - check for similar prompts
            if self.enable_summary_cache:
                # Create a simple hash of the first 300 chars of prompt
                prompt_key = prompt[:300]
                prompt_hash = hashlib.md5(prompt_key.encode()).hexdigest()

                # Check cache
                if prompt_hash in self._summary_cache:
                    cached_summary = self._summary_cache[prompt_hash]
                    if self.logger_manager:
                        self.logger_manager.get_logger('llm_api').debug(
                            f"Using cached summary: {cached_summary}"
                        )
                    return cached_summary

            # Generate new summary
            summary_prompt = f"""Summarize this LLM API call in 1-2 sentences (max 100 chars):

Stage: {stage}
Task: {task}
Prompt (first 500 chars): {prompt[:500]}...
Response (first 500 chars): {response[:500]}...

Focus on: What was the main purpose and outcome?"""

            # Create a temporary LLM client for summarization (use fast/cheap model)
            # Use same API key but different model for speed
            summary_model = 'glm-4-flash' if 'glm' in self.model_name else self.model_name

            # Make API call for summary
            summary_response = self.client.chat.completions.create(
                model=summary_model,
                messages=[
                    {'role': 'system', 'content': 'You are a concise summarizer. Max 100 characters.'},
                    {'role': 'user', 'content': summary_prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused summaries
                max_tokens=100
            )

            summary = summary_response.choices[0].message.content.strip()

            # CRITICAL FIX: Enforce length limit and format
            # Replace newlines with spaces to keep brief_summary as single line
            summary = summary.replace('\n', ' ').replace('\r', ' ')
            # Remove multiple spaces
            import re
            summary = re.sub(r'\s+', ' ', summary)
            # Truncate to 150 characters (with ellipsis if truncated)
            if len(summary) > 150:
                summary = summary[:147] + "..."
            summary = summary.strip()

            # FEATURE 2: Save to cache
            if self.enable_summary_cache:
                prompt_hash = hashlib.md5(prompt[:300].encode()).hexdigest()
                self._summary_cache[prompt_hash] = summary
                # Limit cache size to prevent memory issues
                if len(self._summary_cache) > 100:
                    # Remove oldest entry (FIFO)
                    oldest_key = next(iter(self._summary_cache))
                    del self._summary_cache[oldest_key]

            # Save to latent_summaries.jsonl in debug directory
            if self.logger_manager:
                debug_dir = self.logger_manager.debug_dir
                summary_file = debug_dir / "latent_summaries.jsonl"

                summary_entry = {
                    "timestamp": time.time(),
                    "stage": stage,
                    "task": task,
                    "model": self.model_name,
                    "prompt_tokens": usage_data.get('prompt_tokens', 0),
                    "completion_tokens": usage_data.get('completion_tokens', 0),
                    "summary": summary,
                    "cached": False  # Track if this was cached
                }

                with open(summary_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(summary_entry, ensure_ascii=False) + '\n')

            return summary

        except Exception as e:
            # If summarization fails, just log warning (don't crash the pipeline)
            if self.logger_manager:
                self.logger_manager.get_logger('llm_api').warning(f"Failed to generate latent summary: {e}")
            return None
