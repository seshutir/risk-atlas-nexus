import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from policy_distillation.models.guardians.judge import Judge


class GraniteGuardian(Judge):

    def __init__(self, model_path, config):
        """
        Wrapper class for Granite Guardian
        :param model_path: granite guardian path in a HuggingFace format (e.g. 'ibm-granite/granite-guardian-3.1-2b')
        :param config: a set of parameters describing the task and the criterion
        """
        super().__init__(config)
        self.guardian_model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.guardian_tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.name = "gg"

    def predict(self, prompt=None, response=None):
        label_prediction = self.ask_guardian((prompt, response))
        return self.label_names.index(label_prediction)

    def parse_output(self, output):
        nlogprobs = 20

        if nlogprobs > 0:

            list_index_logprobs_i = [
                torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                for token_i in list(output.scores)[:-1]
            ]
            if list_index_logprobs_i is not None:
                prob = self.get_probabilities(list_index_logprobs_i)

        return prob

    def get_probabilities(self, logprobs):
        safe_token = "No"
        unsafe_token = "Yes"
        nlogprobs = 20

        safe_token_prob = 1e-50
        unsafe_token_prob = 1e-50
        for gen_token_i in logprobs:
            for logprob, index in zip(
                gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]
            ):
                decoded_token = self.guardian_tokenizer.convert_ids_to_tokens(index)
                if decoded_token.strip().lower() == safe_token.lower():
                    safe_token_prob += math.exp(logprob)
                if decoded_token.strip().lower() == unsafe_token.lower():
                    unsafe_token_prob += math.exp(logprob)

        probabilities = torch.softmax(
            torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]),
            dim=0,
        )

        return probabilities

    def ask_guardian(self, message):
        if (isinstance(message, tuple) or isinstance(message, list)) and len(
            message
        ) == 2:
            prompt = message[0]
            response = message[1]
        else:
            prompt = message
            response = None

        messages = [{"role": "user", "content": prompt.strip()}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})

        guardian_config = {"risk_name": self.criterion}
        input_ids = self.guardian_tokenizer.apply_chat_template(
            messages,
            guardian_config=guardian_config,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.guardian_model.device)
        input_len = input_ids.shape[1]

        self.guardian_model.eval()

        with torch.no_grad():
            output = self.guardian_model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )

        safe_token = "No"
        unsafe_token = "Yes"
        res = self.guardian_tokenizer.decode(
            output.sequences[:, input_len:][0], skip_special_tokens=True
        ).strip()
        if unsafe_token.lower() == res.lower():
            label = unsafe_token
        elif safe_token.lower() == res.lower():
            label = safe_token
        else:
            label = "Failed"

        guardian_response = "harmful" if label == "Yes" else "harmless"
        return guardian_response

    def predict_proba(self, inputs):
        probs = []

        guardian_config = {"risk_name": self.criterion}

        for i in inputs:
            with torch.no_grad():
                if isinstance(i, tuple):
                    prompt, response = i
                    messages = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                else:
                    messages = [{"role": "user", "content": i}]

                input_ids = self.guardian_tokenizer.apply_chat_template(
                    messages, guardian_config=guardian_config, return_tensors="pt"
                ).to(self.guardian_model.device)

                input_len = input_ids.shape[1]

                output = self.guardian_model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=20,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            p = self.parse_output(output)
            probs.append(p)

        return np.array(probs)
