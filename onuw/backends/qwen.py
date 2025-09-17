from typing import Dict
import os
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .base import IntelligenceBackend

try:
    from openai import OpenAI
except ImportError:
    is_openai_available = False
else:
    base_url = os.environ.get("QWEN_BASE")
    api_key = os.environ.get("QWEN_API_KEY")
    if base_url is None or api_key is None:
        is_openai_available = False
    else:
        is_openai_available = True

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "Qwen/Qwen3-14B"

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token


class Qwen(IntelligenceBackend):
    """
    Interface to the Qwen model while using the OpenAI API compatible endpoint.
    """
    stateful = False
    type_name = "qwen"

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 model: str = DEFAULT_MODEL, merge_other_agents_as_one_user: bool = True, **kwargs):
        """
        instantiate the Qwen backend
        args:
            temperature: the temperature of the sampling
            max_tokens: the maximum number of tokens to sample
            model: the model to use
            merge_other_agents_as_one_user: whether to merge messages from other agents as one user message
        """
        assert is_openai_available, "openai package is not installed or the API key is not set"
        super().__init__(temperature=temperature, max_tokens=max_tokens, model=model,
                         merge_other_agents_as_one_user=merge_other_agents_as_one_user, **kwargs)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

        self.client = OpenAI(base_url=base_url, api_key=api_key)
    
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages, *args, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=STOP,
            extra_body={"enable_thinking": False}
        )
        response = completion.choices[0].message
        return response
    
    def query(self, agent_name: str, prompts: Dict[str, str], request_msg: str = None, *args, **kwargs) -> str:
        print("Using backend with :" + DEFAULT_MODEL)
        
        # Construct the prompts for ChatGPT
        messages = [
            {"role": "system", "content": prompts.get("system_prompt", "")},
            {"role": "user", "content": prompts.get("user_prompt", "")}
        ]

        # Specific action and desired JSON response format
        if request_msg:
            messages.append({"role": "system", "content": f"{request_msg}/no_think"})
        else:  # The default request message that reminds the agent its role and instruct it to speak
            messages.append({"role": "system", "content": f"Now it is your turn, {agent_name}./no_think"})
        
        # Generate response
        response = self._get_response(messages, *args, **kwargs)

        # Post-process, remove the agent name if the response starts with it
        response = response.content.strip()
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()
        response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()

        return response
