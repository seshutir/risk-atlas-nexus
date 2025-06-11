import jsonschema


# Validate instance data against a json schema
def validate(instance, schema):
    try:
        jsonschema.validate(instance, schema=schema)
        return True
    except:
        return False
