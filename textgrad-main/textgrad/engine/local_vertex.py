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
            config_path: str = "config.json",
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

    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str = None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        elif isinstance(content, list):
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
            self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=8192, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt}
        ]

        response = completion(
            model=self.model_string,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            vertex_project=self.vertex_project_id,
            vertex_location=self.vertex_location
        )

        response_text = response.choices[0].message.content
        return response_text

    def _generate_from_multiple_input(
            self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=8192, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        responses = []

        # Process each content item separately
        for item in content:
            if not isinstance(item, str):
                raise ValueError("Multimodal input is not supported for Vertex AI models")

            # Create messages for this single item
            messages = [
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": item}
            ]

            response = completion(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                vertex_project=self.vertex_project_id,
                vertex_location=self.vertex_location
            )

            responses.append(response.choices[0].message.content)

        return responses

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)
