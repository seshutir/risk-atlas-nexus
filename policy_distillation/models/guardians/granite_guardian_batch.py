import math

import numpy as np
import torch
from transformers import AutoTokenizer

from policy_distillation.llms.rits_batch_component import RITSBatchModel
from policy_distillation.models.guardians.judge import Judge


class GGRits(Judge):

    def __init__(self, model_path, config):
        super().__init__(config)
        rits_models = ['ibm-granite/granite-guardian-3.2-5b', 'ibm-granite/granite-guardian-3.2-3b-a800m']
        rits_model_names = ['granite-guardian-3-2-5b-ris', 'granite-guardian-3-2-3b-a800m']

        i = rits_models.index(model_path)
        rits_config = {
                "base-url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com",
                "num-parallel-requests": 10,
                "max-retries": 10,
                "temperature": 1.0,
                "max-new-tokens": 150,
                "min-new-tokens": 10,
                "top-p": 0.95,
                "do-sample": True,
                "output-format": [
                    {
                        "name": "answer",
                        "description": "answer for the question"
                    }
                ]
            }
        output_labels = config["output_labels"]
        self.guardian_model = RITSBatchModel(model_name=model_path,
                                             model_path=rits_model_names[i],
                                            output_labels = output_labels,
                                             config=rits_config)

        self.guardian_tokenizer = AutoTokenizer.from_pretrained(model_path)

    def ask_guardian(self, message):
        if (isinstance(message, tuple) or isinstance(message, list)) and len(message) == 2:
            prompt = message[0]
            response = message[1]
        else:
            prompt = message
            response = None

        messages = [{"role": "user", "content": prompt, "name": "test"}]
        if response is not None:
            messages.append({"role": "assistant", "content": response, "name": "test"})

        responses = self.guardian_model.generate(messages)
        harmful = responses[0]['answer'] == 'Yes'
        return 'harmful' if harmful else 'harmless'

    def predict_proba(self, inputs):
        messages = []
        for i in inputs:
            messages.append({"role": "user", "content": i.strip()})

        responses = self.guardian_model.generate(messages)
        probs = []
        for p in [r['probs'] for r in responses]:
            probs.append(p)

        return np.array(probs)
