import inspect

from ChannelContrib.carryover import ExponentialCarryover,GaussianCarryover
from ChannelContrib.saturation import ExponentialSaturation,BoxCoxSaturation,HillSaturation


# Step 1: Map function names to their classes
carryover_functions = {
    "ExponentialCarryover": ExponentialCarryover,
    "GaussianCarryover": GaussianCarryover,
}


# Function to extract parameter names
def get_carryover_function_parameter_names(function_name):
    # Step 2: Get the class from the dictionary
    function_class = carryover_functions.get(function_name)
    if not function_class:
        return "Function name not found."

    # Step 3 & 4: Use inspect to get the parameters of the __init__ method
    parameters = inspect.signature(function_class.__init__).parameters

    # Step 5: Extract parameter names, excluding 'self'
    parameter_names = [name for name in parameters if name != 'self']

    return parameter_names


# Example usage




# Step 1: Map saturation model names to their classes
saturation_models = {
    "ExponentialSaturation": ExponentialSaturation,
    "BoxCoxSaturation": BoxCoxSaturation,
    "HillSaturation": HillSaturation,
    # "AdbudgSaturation": AdbudgSaturation,
}


# Function to extract parameter names
def get_saturation_model_parameter_names(model_name):
    # Step 2: Get the class from the dictionary
    model_class = saturation_models.get(model_name)
    if not model_class:
        return "Model name not found."

    # Use inspect to get the parameters of the __init__ method
    parameters = inspect.signature(model_class.__init__).parameters

    # Step 3: Extract parameter names, excluding 'self'
    parameter_names = [name for name in parameters if name != 'self']

    return parameter_names


# Example usage

