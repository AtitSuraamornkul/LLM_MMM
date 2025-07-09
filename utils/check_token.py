import tiktoken

def num_tokens_from_string(string: str, model_name: str = 'gpt-3.5-turbo') -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))




def count_message_tokens(messages, model_name="gpt-3.5-turbo"):
    total = 0
    for msg in messages:
        total += num_tokens_from_string(msg["content"], model_name)
    return total


