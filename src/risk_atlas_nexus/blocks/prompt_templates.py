QUESTIONNAIRE_COT_TEMPLATE = """
        I want you to play the role of a compliance officer and answer the question based on the given Intent.
        Return the question, answer and explanation in a json format where question, answer and explanation are keys of the json exactly as shown in the examples.
        you should answer the question followed by an explanation on how that answer was generated.
{% if cot_examples is not none %}{% for example in cot_examples %}
        Intent: {{ example.intent }}
        Question: {{ question }}
        Answer: {{ example.answer }}
{% if example.explanation is not none %}        Explanation: {{ example.explanation }}{% endif %}
{% endfor %}{% endif %}
        Intent: {{ usecase }}
        Question: {{ question }}
"""

RISK_IDENTIFICATION_TEMPLATE = """You are an expert at AI risk classification. Study the risks JSON below containing list of risk category and its description.

RISKS:
{{ risks }}

Instructions:
1. Identify the potential RISKS associated with the given Input. Use RISK `description` to verify if the risk is associated with the Input.
2. If Input doesn't fit into any of the above RISKS categories, classify it as Unknown.
3. Respond with a list{% if max_risk is not none %} (top {{ max_risk }} high risks categories){% endif %} of attribute 'category' containing the risk labels.
{% if cot_examples is not none and cot_examples|length > 0 %}
EXAMPLES:{% for example in cot_examples %}
Input: {{ example.Input }}
Output: {{ example.Output }}{% endfor %}
===== END OF EXAMPLES ======
{% endif %}
Input: {{ usecase }}
Output: """

AI_TASKS_TEMPLATE = """Study and understand the JSON below containing a list of LLM task and its description.

{{ hf_ai_tasks }}

Your task is to identify one or more LLM tasks for the context given below. Respond only with a JSON list (maximum length {{ limit }} items) containing the most relevant LLM task labels. Do not include description. Ensure that your answer only includes the json list.

Context: {{ usecase }}
Output: """
