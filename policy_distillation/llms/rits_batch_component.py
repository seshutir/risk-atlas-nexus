import math
import os

import requests
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


class RITSBatchModel:
    def __init__(self, model_name, model_path, output_labels, config, endpoint='v1/chat/completions'):
        self.model_name = model_name
        self.model_path = model_path
        self.output_labels = output_labels

        self.config = config

        # params
        self.num_parallel_requests = self.config['num-parallel-requests']

        # API key
        load_dotenv()

        # client
        self.model_name = model_name
        self.endpoint = os.path.join(self.config['base-url'], model_path, endpoint)
        # output parser
        self.parser = StructuredOutputParser.from_response_schemas(self.config['output-format'])
        self.format_instructions = self.parser.get_format_instructions()

    def generate(self, prompts):
        responses = [None] * len(prompts)
        retry_indices = list(range(len(prompts)))
        retry_count = 0

        original_prompts = prompts

        total_attempts = 0
        with tqdm(total=len(prompts), desc="Generating") as pbar:
            while retry_indices and retry_count < self.config['max-retries']:
                retry_mapping = {i: original_idx for i, original_idx in enumerate(retry_indices)}

                prompts = self._filter_prompts(original_prompts, retry_indices)

                generation_params = self._get_generation_params()
                args_list = [(prompt, generation_params) for prompt in prompts]

                generation_outputs = thread_map(
                    self._get_rits_completion,
                    args_list,
                    max_workers=self.num_parallel_requests,
                )
                generation_outputs = list(generation_outputs)

                prev_retry_count = len(retry_indices)
                retry_indices, responses = self._process_outputs(generation_outputs, responses, retry_mapping)

                successful_generations = prev_retry_count - len(retry_indices)
                pbar.update(successful_generations)

                retry_count += 1
                total_attempts += len(prompts)

                success_rate = ((len(prompts) - len(retry_indices)) / total_attempts) * 100
                pbar.set_description(
                    f"progress: {len(prompts) - len(retry_indices)}/{len(prompts)} "
                    f"(parsing success rate: {success_rate:.1f}%)"
                )

        return [resp if resp is not None else {} for resp in responses]

    def _get_rits_completion(self, args):
        prompt, generation_params = args
        if not isinstance(prompt, list):
            prompt = [prompt]
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "RITS_API_KEY": os.getenv('RITS_API_KEY'),
                    "Content-Type": "application/json"},
                json={
                    "model": self.model_name,
                    "messages": prompt,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 20
                }
            )

            answer = response.json()['choices'][0]['message']['content'].strip().split('\n')[0]
            if answer.strip() == self.output_labels[0]:
                no_log_prob = response.json()['choices'][0]['logprobs']['content'][0]['top_logprobs'][0]['logprob']
                no_prob = math.e**no_log_prob
                yes_prob = 1 - no_prob
            elif answer.strip() == self.output_labels[1]:
                yes_log_prob = response.json()['choices'][0]['logprobs']['content'][0]['top_logprobs'][0]['logprob']
                yes_prob = math.e ** yes_log_prob
                no_prob = 1 - yes_prob
            else:
                raise ValueError(f'Output {answer} does not match any of the output labels {self.output_labels}')

            return {"answer": answer.strip(),
                    "probs": [no_prob, yes_prob]}

        except Exception as e:
            return None

    def _process_outputs(self, generation_outputs, responses, retry_mapping):
        new_retry_indices = []

        for temp_idx, temp_response in enumerate(generation_outputs):
            original_idx = retry_mapping[temp_idx]

            if temp_response is None:
                new_retry_indices.append(original_idx)
                responses[original_idx] = None
            else:
                responses[original_idx] = temp_response

        return new_retry_indices, responses

    @staticmethod
    def _filter_prompts(prompts, indices):
        return [
            prompts[i] for i in indices
        ]

    def _get_generation_params(self):
        params = {
            'temperature': 0.0,
            'max_new_tokens': 100,
            'min_new_tokens':  10,
            'top_p': 0.95,
            'do_sample':True
        }
        return params
