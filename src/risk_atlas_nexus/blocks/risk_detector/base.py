from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Container,
    Risk,
    RiskTaxonomy,
)
from risk_atlas_nexus.blocks.inference.base import InferenceEngine
from risk_atlas_nexus.blocks.prompt_builder import (
    FewShotPromptBuilder,
    ZeroShotPromptBuilder,
)
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.toolkit.logging import configure_logger
from risk_atlas_nexus.toolkit.validate import validate


LOGGER = configure_logger(__name__)


RISK_IDENTIFICATION_EXAMPLES = load_resource("risk_generation_cot.json")
RISK_IDENTIFICATION_COT_SCHEMA = load_resource("risk_generation_cot_schema.json")


class RiskDetector(ABC):

    def __init__(
        self,
        ontology: Container,
        inference_engine: InferenceEngine,
        taxonomy: Optional[str] = None,
        cot_examples: Optional[Dict[str, List]] = None,
        max_risk: Optional[int] = None,
    ):
        self.inference_engine = inference_engine
        self._ontology = ontology
        self._taxonomy_id = taxonomy if taxonomy else "ibm-risk-atlas"
        self._risks = self.get_risks_by_taxonomy_id(ontology, self._taxonomy_id)

        # Validate user-provided `cot_examples` format if available.
        if not validate(cot_examples, RISK_IDENTIFICATION_COT_SCHEMA):
            raise Exception(
                f"The format of `cot_examples` is incorrect. Please refer to the example template provided at src/risk_atlas_nexus/data/templates/risk_generation_cot.json"
            )

        # First, check if the user has provided 'cot_examples'. If not,
        # retrieve the default cot examples from the master. If no examples
        # exist in the master, set it as None.
        self._examples = (
            cot_examples and cot_examples.get(self._taxonomy_id, None)
        ) or RISK_IDENTIFICATION_EXAMPLES.get(self._taxonomy_id, None)

        # Set prompt builder based on whether the CoT examples are available.
        if self._examples is None:
            LOGGER.warning(
                f"Warning: Chain of Thought (CoT) examples were not provided, or do not exist in the master for the taxonomy type: {self._taxonomy_id}. The API will use the Zero shot method. To improve the accuracy of risk identification, please provide CoT examples in `cot_examples` when calling this API. You may also consider raising an issue to permanently add these examples to the Risk Atlas Nexus master."
            )
            self.prompt_builder = ZeroShotPromptBuilder
        else:
            self.prompt_builder = FewShotPromptBuilder

        self.max_risk = max_risk

    def get_risks_by_taxonomy_id(
        self, ontology: Container, taxonomy_id: Optional[str] = None
    ):
        taxonomies = list(
            filter(
                lambda taxonomy: taxonomy.id == taxonomy_id,
                ontology.taxonomies,
            )
        )

        if len(taxonomies) > 0:
            taxonomy: RiskTaxonomy = taxonomies[0]
            LOGGER.info(
                f"Selected taxonomy is {str(taxonomy.name)}. For more info: {taxonomy.url}"
            )

            return list(
                filter(
                    lambda risk: risk.isDefinedByTaxonomy == taxonomy.id,
                    ontology.risks,
                )
            )
        else:
            raise Exception(
                f"Risk Taxonomy: {taxonomy_id} not found. Available taxonomies: {[taxonomy.id for taxonomy in ontology.taxonomies]}"
            )

    @abstractmethod
    def detect(self, usecases: List[str]) -> List[List[Risk]]:
        raise NotImplementedError
