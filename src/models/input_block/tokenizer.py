from tiktoken.load import load_tiktoken_bpe
from tiktoken import Encoding, get_encoding


class LlamaTokenizer: 
    
    def __init__(self, model_name: str = "cl100k_base"): 
        # get vocab size of tokennizer base model
        model_base = get_encoding(model_name)
        
        vocab_size = len(model_base._mergeable_ranks) + len(model_base._special_tokens)
        
        # regex để tách từ 
        pat_str =  r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" 
        
        # setup special tokens 
        special_tokens = ['<|beginoftext|>','<|pad_id|>']
        self.bot_token = special_tokens[0]
        self.pad_token = special_tokens[1]
        self.eot_token = '<|endoftext|>'
        
        self.special_tokens = model_base._special_tokens
        self.special_tokens.update({
            tokens : i + vocab_size for i, tokens in enumerate(special_tokens) 
        })
        
        self.model = Encoding(
            name = model_name, 
            pat_str= pat_str,
            special_tokens = self.special_tokens, 
            mergeable_ranks= model_base._mergeable_ranks
        )
        
    def encode(self, text: str, allow_special: bool = True): 
        allowed_special = set(self.special_tokens.keys()) if allow_special else set()
        return self.model.encode(text, allowed_special=allowed_special)
    
    def decode(self, tokens: list, allow_special: bool = True): 
        allowed_special = set(self.special_tokens.keys()) if allow_special else set()
        return self.model.decode(tokens, allowed_special=allowed_special)


if __name__ == "__main__": 
      
    def test_tokenizer(): 
        tokenizer = LlamaTokenizer()
        
        print("INIT PASSED")
        text = "Hello, this is a test"
        tokens = tokenizer.encode(text)
        print(tokens)
        print('ENCODE PASSED')
        
        decoded_text = tokenizer.decode(tokens)
        print(decoded_text)
        print("DECODE PASSED")
        
    test_tokenizer()
    
    