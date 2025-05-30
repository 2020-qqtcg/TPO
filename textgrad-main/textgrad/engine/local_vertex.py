import os
import json
from typing import Union, List
from litellm import completion
from .base import EngineLM, CachedEngine


class LocalVertex(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
            self,
            model_string: str = "vertex_ai/gemini-pro",
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            config_path: str = r"C:\Users\ufo\zooptimization\optimization\config\vertex_config.json",
            **kwargs):
        """
        Initialize LocalVertex with Vertex AI configuration from config file
        
        :param model_string: The model to use (default: vertex_ai/gemini-pro)
        :param system_prompt: System prompt for the model
        :param config_path: Path to the config.json file containing Vertex AI settings
        """
        root = os.path.expanduser("~/.cache/textgrad")
        cache_path = os.path.join(root, f"cache_vertex_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.model_string = model_string

        # Load configuration from config file
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Set up Vertex AI credentials from config
            if credential_path := config.get("credential_path"):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credential_path
                os.environ["VERTEX_PROJECT"] = config.get("vertex_project_id")
                os.environ["VERTEX_LOCATION"] = config.get("vertex_location")

            self.vertex_project_id = config.get("vertex_project_id")
            self.vertex_location = config.get("vertex_location")

            if not self.vertex_project_id or not self.vertex_location:
                raise ValueError("vertex_project_id and vertex_location must be specified in config.json")

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file at {config_path}")

    def generate(
        self, prompt, system_prompt=None, temperature=0.7, max_tokens=4096, top_p=0.95, n=1, **kwargs
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        messages = []
        if sys_prompt_arg:
            messages.append({"role": "system", "content": sys_prompt_arg})
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # top_p=top_p,
            n=n,
            vertex_project=self.vertex_project_id,
            vertex_location=self.vertex_location
        )

        if n > 1:
            return [choice.message.content for choice in response.choices]
        else:
            response_text = response.choices[0].message.content
            return response_text

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
