from dotenv import load_dotenv
from risk_policy_distillation.datasets.prompt_response_dataset import PromptResponseDataset
from datasets import load_dataset

from src.risk_atlas_nexus import RiskAtlasNexus


def main():
    load_dotenv()

    # get the dataset
    ds = load_dataset("PKU-Alignment/BeaverTails")
    dataframe = ds.data['330k_train'].table.to_pandas()

    # sample for testing purposes
    dataframe = dataframe.sample(10)

    # provide some information in the data config
    data_config = {
        "dataset_name": "BeaverTails",
        "index_col": "", # if there is no index column it will be automatically populated
        "prompt_col": "prompt",
        "response_col": "response",
        "label_col": "is_safe",
    }

    dataset = PromptResponseDataset(dataframe=dataframe, config=data_config)

    # provide a guardian config
    guardian_config = {
        "task": f"Harm detection",
        "criterion": f"Harm",
        "criterion_definition": f"The message has harmful by common-sense.",
        "labels": [0, 1],
        "label_names": ['harmless', 'harmful'],
        "output_labels": ["no", "yes"]
    }

    risk_atlas_nexus = RiskAtlasNexus()

    expl = risk_atlas_nexus.generate_policy_rules(guardian_config, dataset, local_expl='lime', results_folder='results')

    print(expl.print())



if __name__ == '__main__':
    main()