
from dotenv import load_dotenv
from risk_atlas_nexus.library import RiskAtlasNexus


def main():
    load_dotenv()

    # Creating a test dataset
    bt_config = {
        "general": {
            "location": "PKU-Alignment/BeaverTails",
            "dataset_name": "BeaverTails"
        },
        "data": {
            "type": "prompt_response",
            "index_col": "",
            "prompt_col": "prompt",
            "response_col": "response",
            "label_col": "is_safe",
            "flip_labels": True,
            "category_label": "category_simple"
        },
        "split": {
            "split": False,
            "sample_ratio": 0.0001,
            "subset": "330k_train"
        }
    }

    risk_atlas_nexus = RiskAtlasNexus()

    expl = risk_atlas_nexus.generate_policy_rules(task="harm detection", label_names=["harmless", "harmful"],
                                                  dataset_config=bt_config)

    print(expl)



if __name__ == '__main__':
    main()