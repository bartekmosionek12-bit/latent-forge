import time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache
torch.manual_seed(0)
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=torch.float32).eval()
m.requires_grad_(False)
P,S,Q,T=250,20,45,60   # page prefix, suffix, post-suffix (page end+question+template), target continuation
pre=torch.randint(100,20000,(1,P))
with torch.no_grad(): pkv=m(pre,use_cache=True).past_key_values
def cache(B):
    c=DynamicCache()
    for l in range(len(pkv)):
        c.update(pkv[l][0].expand(B,-1,-1,-1), pkv[l][1].expand(B,-1,-1,-1), l)
    return c
def eval_cands(ncand,k,chunk=32,upto=None):
    ids=torch.randint(100,20000,(ncand*k,S+Q+T)); t=time.time()
    with torch.no_grad():
        for i in range(0,ids.shape[0],chunk):
            x=ids[i:i+chunk]; B=x.shape[0]
            pos=torch.arange(P,P+x.shape[1]).unsqueeze(0).expand(B,-1)
            if upto is None:
                h=m.model(input_ids=x,past_key_values=cache(B),position_ids=pos).last_hidden_state
                lg=m.lm_head(h[:,S+Q-1:-1]); F.cross_entropy(lg.reshape(-1,lg.shape[-1]),x[:,S+Q:].reshape(-1))
                m.lm_head(h[:,:S])  # fluency logits at suffix positions
            else:
                layers=m.model.layers; m.model.layers=layers[:upto+1]
                m.model(input_ids=x[:,:S+Q],past_key_values=cache(B),position_ids=pos[:,:S+Q])
                m.model.layers=layers
    return time.time()-t
def grad(k):
    ids=torch.randint(100,20000,(k,S+Q+T)); E=m.get_input_embeddings().weight
    oh=F.one_hot(ids[:,:S],E.shape[0]).float().requires_grad_(True)
    t=time.time()
    emb=torch.cat([oh@E, m.get_input_embeddings()(ids[:,S:])],1)
    pos=torch.arange(P,P+ids.shape[1]).unsqueeze(0).expand(k,-1)
    h=m.model(inputs_embeds=emb,past_key_values=cache(k),position_ids=pos).last_hidden_state
    lg=m.lm_head(h[:,S+Q-1:-1]); F.cross_entropy(lg.reshape(-1,lg.shape[-1]),ids[:,S+Q:].reshape(-1)).backward()
    return time.time()-t
for k in (1,4,8):
    g=grad(k); e=eval_cands(128,k); print(f"continuation target, k={k} targets/step, 128 cand: grad {g:.1f}s + eval {e:.1f}s = {g+e:.1f}s/step", flush=True)
e=eval_cands(128,1,upto=8); print(f"activation band to layer 8, 128 cand: eval {e:.1f}s (+grad ~similar to k=1 truncated)", flush=True)
