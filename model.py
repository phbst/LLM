
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim:int=288
    n_layers:int=6
    n_heads:int=6
    n_kv_heads:int=6
    vocab_size:int=32000
    hidden_dim:int=768
    norm_eps:float=1e-5
    max_seq_len:int=256
    dropout:float=0.0


class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float):
        super().__init__()
        self.eps = eps
        self.weight=nn.Parameter(torch.ones(dim))  # w与b


    def forward(self, x:torch.Tensor):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps).type_as(x)*self.weight


def compute_freqs_sc(dim:int,len_sc:int):
    freqs=1/(10000**(torch.arange(0,dim,2)[:(dim//2)].float()/dim)) #token内部
    len_freqs = torch.arange(len_sc,device=freqs.device).float() #seqence len
    freqs_sc=torch.outer(len_freqs,freqs).float()
    freqs_cos= torch.cos(freqs_sc)
    freqs_sin= torch.sin(freqs_sc)
    return freqs_cos,freqs_sin


def reshape_for_broadcast(freqs_cis,x):  # x(bs,seq_len,head,dim)

    assert freqs_cis.shape==(x.shape[1],x.shape[-1])
    shape=[1,x.shape[1],1,x.shape[-1]]
    return freqs_cis.view(shape)

def apply_rope(xq,xk,freqs_cos,freqs_sin):
    xq_r,xq_i=xq.float().reshape(xq.shape[:-1]+(-1,2)).unbind(-1)
    xk_r,xk_i=xk.float().reshape(xk.shape[:-1]+(-1,2)).unbind(-1)

    freqs_cos=reshape_for_broadcast(freqs_cos,xq_r)
    freqs_sin=reshape_for_broadcast(freqs_sin,xq_r)
    
    xq_out_r=xq_r*freqs_cos-xq_i*freqs_sin
    xq_out_i=xq_r*freqs_sin+xq_i*freqs_cos
    xk_out_r=xk_r*freqs_cos-xk_i*freqs_sin
    xk_out_i=xk_r*freqs_sin+xk_i*freqs_cos
    
    xq_out=torch.stack([xq_out_r,xq_out_i],dim=-1).flatten(3)
    xk_out=torch.stack([xk_out_r,xk_out_i],dim=-1).flatten(3)

    return xq_out.type_as(xq),xk_out.type_as(xk)

def repeat_kv(kv:torch.Tensor,n:int):
    bs,seqlen,kv_head,dim=kv.shape
    if n==1:return kv
    return kv[:,:,:,None,:].expand([bs,seqlen,kv_head,n,dim]).reshape([bs,seqlen,kv_head*n,dim])



class Attention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        # dim:int=288
        # n_layers:int=6
        # n_heads:int=6
        # n_kv_heads:int=6
        # vocab_size:int=32000
        # hidden_dim:int=None
        # norm_eps:float=1e-5
        # max_seq_len:int=256
        # dropout:float=0.0
        self.dim=args.dim
        self.n_heads=args.n_heads
        self.n_kv_heads=args.n_kv_heads
        self.n_kv_repeat=self.n_heads//self.n_kv_heads
        self.max_seq_len=args.max_seq_len
        self.head_dim=self.dim//self.n_heads
        self.dropout=args.dropout

        mask = torch.full((1, 1,self.max_seq_len,self.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)
        
        self.wq=nn.Linear(self.dim,self.dim, bias=False)
        self.wk=nn.Linear(self.dim,self.n_kv_heads*self.head_dim, bias=False)
        self.wv=nn.Linear(self.dim,self.n_kv_heads*self.head_dim, bias=False)

        self.wo=nn.Linear(self.dim,self.dim, bias=False)

        self.atte_droupout=nn.Dropout(self.dropout)
        self.resid_dropout=nn.Dropout(self.dropout)
    
    def forward(self,x,freqs_cos,freqs_sin):
        bs,seqlen,dim=x.shape
        xq,xk,xv=self.wq(x),self.wk(x),self.wv(x)

        xq=xq.view(bs,seqlen,self.n_heads,self.head_dim)
        xk=xk.view(bs,seqlen,self.n_kv_heads,self.head_dim)
        xv=xv.view(bs,seqlen,self.n_kv_heads,self.head_dim)

        xq,xk=apply_rope(xq,xk,freqs_cos,freqs_sin)
        xk=repeat_kv(xk,self.n_kv_repeat)
        xv=repeat_kv(xv,self.n_kv_repeat)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2) #(bs,seqlen,n_kv_heads,n_kv_heads)
        xv = xv.transpose(1, 2)


        score=torch.matmul(xq,xk.transpose(-2,-1))/torch.sqrt(torch.tensor(self.head_dim))
        score=score+self.mask[:,:,:seqlen,:seqlen]
        score=F.softmax(score,dim=-1).type_as(xq)
        score=self.atte_droupout(score)
        out=torch.matmul(score,xv)
        out=out.transpose(1, 2).contiguous().view(bs,seqlen,-1)
        output=self.wo(out)
        output=self.resid_dropout(output)
        return output  

class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,dropout):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim,self.hidden_dim,bias=False)

        self.dropout = nn.Dropout(self.dropout)
    def forward(self,x):

        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self,args:ModelArgs,layer_id):
        super().__init__()
        self.dim=args.dim
        self.hidden_dim=args.hidden_dim
        self.dropout=args.dropout
        self.norm_eps=args.norm_eps
        self.attention=Attention(args)
        self.feed_forward=MLP(self.dim,self.hidden_dim,self.dropout)
        self.layer_id=layer_id
        self.attention_norm = RMSNorm(self.dim, eps=self.norm_eps)
        self.fnn_norm=RMSNorm(self.dim, eps=self.norm_eps)
    
    def forward(self,x,freqs_cos,freqs_sin):
        h=x+self.attention(self.attention_norm(x),freqs_cos,freqs_sin)
        out=h+self.feed_forward.forward(self.fnn_norm(h))
        return out



class Bobllm(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        #     dim:int=288
        # n_layers:int=6
        # n_heads:int=6
        # n_kv_heads:int=6
        # vocab_size:int=32000
        # hidden_dim:int=None
        # norm_eps:float=1e-5
        # max_seq_len:int=256
        # dropout:float=0.0
    
        self.args=args
        self.n_layers=args.n_layers
        self.vocab_size=args.vocab_size
        self.hidden_dim=args.hidden_dim
        self.dim=args.dim
        self.dropout=args.dropout
        self.norm_eps=args.norm_eps
        self.max_seq_len=args.max_seq_len
        self.embdedding=nn.Embedding(self.vocab_size,self.dim)
        self.dropout=nn.Dropout(self.dropout)
        self.layers=nn.ModuleList([DecoderLayer(args,i) for i in range(self.n_layers)])
        self.norm=RMSNorm(self.dim, eps=self.norm_eps)
        self.output=nn.Linear(self.dim,self.vocab_size,bias=False)
        self.embdedding.weight=self.output.weight
        head_dim = self.dim // self.args.n_heads
        freqs_cos,freqs_sin=compute_freqs_sc(head_dim,self.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,tokens,targets:Optional[torch.Tensor]):
        
        bs,seq_len=tokens.shape
        h=self.embdedding(tokens)
        h=self.dropout(h)

        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        for layer in self.layers:
            h=layer(h,freqs_cos,freqs_sin)
        h = self.norm(h)
        if targets is not None: 
            logits=self.output(h)
            self.last_loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
        else:
            logits=self.output(h[:,[-1],:])
            self.last_loss=None
        
        return logits

    def generate(self,idx,max_new_tokens=128,temperature=1.0,top_k=None):
        for _ in range(max_new_tokens):
            logits=self(idx,None)
            logits=logits[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(self,weight_decay,lr,betas,device_type):
        param_dict={pn:p for pn,p in self.named_parameters() if p.requires_grad}
        decay_params=[ p for pn,p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for pn,p in param_dict.items() if p.dim()<2]

        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        print(f"Number of parameters: {num_decay_params+num_nodecay_params} ")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer



if __name__ == '__main__':

    args = ModelArgs()
    x = torch.randint(0, 32000, (1, 50))
    model=Bobllm(args)
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)

    out = model(x,None)
    print(out.shape) # [batch_size, 1, vocab_size]
    
    out = model.generate(x)
    print(out.shape)

