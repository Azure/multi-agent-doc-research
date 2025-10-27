import tiktoken

def get_token_numbers(string: str, model_name: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    

