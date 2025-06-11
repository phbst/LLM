
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self,path):
        self.sp_model = SentencePieceProcessor(model_file=path)
        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

    
    def decode(self,text):
        out=self.sp_model.decode(text)
       
        return out
    
    def encode(self,text,bos,eos):
        out=self.sp_model.encode(text)
        if bos:
            out = [self.bos_id]+out
        if eos:
            out = out+[self.eos_id]
        
        return out

if __name__ == "__main__":
    text="hello world, this is a test"
    tokenizer=Tokenizer("/workspace/projects/Bob_llama/tokenizer/tokens_4096.model")
    print(tokenizer.encode(text,bos=True,eos=True))
    tokens=[432,43,754,890,45]
    print(tokenizer.decode(tokens))




