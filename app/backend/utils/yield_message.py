import base64


def send_step_with_code(step_name: str, code: str) -> str:
    """Send a step with code content"""
    encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    return f"### {step_name}#code#{encoded_code}"

def send_step_with_input(step_name: str, description: str) -> str:
    """Send a step with input description"""
    return f"### {step_name}#input#{description}"

def send_step_with_code_and_input(step_name: str, code: str, description: str) -> str:
    """Send a step with both code and input description"""
    encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    return f"### {step_name}#input#{description}#code#{encoded_code}"
