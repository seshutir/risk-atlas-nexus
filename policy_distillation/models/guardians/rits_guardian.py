import numpy as np
from transformers import AutoTokenizer

from policy_distillation.llms.rits_batch_component import RITSBatchModel
from policy_distillation.models.guardians.judge import Judge
from policy_distillation.models.guardians.tokenizer import RITSTokenizer


class RITSGuardian(Judge):

    def __init__(self, model_name, model_served_name, config, name):
        super().__init__(config)

        self.name = name

        self.model_name = model_name
        self.model_served_name = model_served_name

        rits_config = {
            "base-url": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com",
            "num-parallel-requests": 10,
            "max-retries": 100,
            "output-format": [
                {"name": "answer", "description": "answer for the question"}
            ],
        }

        self.guardian_model = RITSBatchModel(
            model_name=model_served_name,
            model_path=model_name,
            output_labels=self.output_labels,
            config=rits_config,
        )

        self.guardian_tokenizer = RITSTokenizer(model_served_name)

    def ask_guardian(self, message):
        if (isinstance(message, tuple) or isinstance(message, list)) and len(
            message
        ) == 2:
            prompt = message[0]
            response = message[1]
        else:
            prompt = message
            response = None

        messages = [{"role": "user", "content": prompt, "name": "test"}]
        if response is not None:
            messages.append({"role": "assistant", "content": response, "name": "test"})

        responses = self.guardian_model.generate([messages])
        output_id = self.output_labels.index(responses[0]["answer"].strip())

        return self.label_names[output_id]

    def predict_proba(self, inputs):
        messages = []
        for i in inputs:
            messages.append({"role": "user", "content": i.strip()})

        responses = self.guardian_model.generate(
            messages
        )  # because this is only used by the likes of LIME, it's ok to parse it as a single text from the user.
        probs = []
        for p in [r["probs"] for r in responses]:
            probs.append(p)

        return np.array(probs)
