"""
This script is used for utility functions
"""
import json


def convert_value(value):
    """
    Convert the string value from the environment variable to the appropriate type.

    Args:
        value: The string value from the environment variable.

    Returns:
        The converted value.
    """
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        int_value = int(value)
        return int_value
    except ValueError:
        pass
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        pass
    try:
        json_value = json.loads(value)
        return json_value
    except (ValueError, json.JSONDecodeError):
        pass
    return value
