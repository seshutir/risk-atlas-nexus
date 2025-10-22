import os

import requests

from policy_distillation.llms.llm_component import LLMComponent


class RITSComponent(LLMComponent):

    def __init__(self, model_name, model_served_name, n_tries=100):
        """
        A wrapper for LLM queries through RITS
        :param model_name: the last part (after the last /) of the "Model inference endpoint" for a RITS model
        :param model_served_name: "Served model name" for a RITS model
        """
        super().__init__()
        self.connection_type = "rits"
        self.model_name = model_name
        self.model_served_name = model_served_name

        self.rits_url = f"https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/{self.model_name}/v1/chat/completions"
        self.n_tries = n_tries

    def send_request(self, context, prompt, response=None, temperature=0.7):
        messages = self.generate_message(context, prompt, response)
        response = self._send_request(messages, temperature)

        return response

    def _send_request(self, messages, temperature=0.7):
        if os.getenv("RITS_API_KEY") is None:
            print("rits key not found")

        i = 0
        while i < self.n_tries:
            response = requests.post(
                self.rits_url,
                headers={
                    "RITS_API_KEY": os.getenv("RITS_API_KEY"),
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_served_name,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            if response.status_code == 200:
                break
            else:
                i += 1

        return response.json()["choices"][0]["message"]["content"]
