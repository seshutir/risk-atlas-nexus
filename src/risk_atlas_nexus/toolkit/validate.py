import jsonschema
from jsonschema.exceptions import ValidationError


# Validate instance data against a json schema
def validate(instance, schema):
    try:
        jsonschema.validate(instance, schema=schema)
    except ValidationError as e:
        return e.message
