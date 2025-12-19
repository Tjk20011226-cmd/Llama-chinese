

from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM ,AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('output_model',use_fast=False)

tokens=['<WJJ>']
tokenizer.add_tokens(tokens)
tokens=['<QJJ>']
tokenizer.add_tokens(tokens)
tokens=['<WLS>']
tokenizer.add_tokens(tokens)
tokens=['<QLS>']
tokenizer.add_tokens(tokens)

tokenizer.save_pretrained('my_tokenizer')
