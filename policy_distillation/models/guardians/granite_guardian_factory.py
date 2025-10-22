from src.models.components.llms.rits_batch_component import RITSBatchModel
from src.models.guardians.granite_guardian import GraniteGuardian


class GGFactory:

    def __init__(self):
        pass

    @staticmethod
    def get_guardian(self, model_path):
        rits_models = ['ibm-granite/granite-guardian-3.2-5b', 'ibm-granite/granite-guardian-3.2-3b-a800m']
        rits_model_names = ['granite-guardian-3-2-5b-ris', 'granite-guardian-3-2-3b-a800m']

        if model_path in rits_models:  # models available on RITS -- this is preferred calls can be batched
            config = {

            }
            i = rits_models.index(model_path)

            self.guardian_model = RITSBatchModel(model_name=model_path,
                                                 model_path=rits_model_names[i],
                                                 config=config)

            return GraniteGuardian(self.guardian)

            self.guardian_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.batch_enabled = True
        else:
            self.batch_enabled = False
            self.guardian_model = self.get_guardian(model_path)
            AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )