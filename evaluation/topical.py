"""INDEPENDENT topical score of generated text, used to check region-targeted attacks.

Independent of the attack's loss on purpose: it never looks at the attacked
model's hidden states. Two components:
  * embedding: a separate sentence-transformer (default all-MiniLM-L6-v2) embeds the
    continuation; cosine distance to the same model's centroid of the reference texts
  * keywords : fraction of the continuation's words found in a hand-written domain list
"""
from __future__ import annotations

import re

import torch

from models import load_embedder

COOKING_KEYWORDS = tuple(sorted(set("""
cook cooking cooked recipe recipes bake baking baked boil boiling fry frying fried roast
roasting roasted simmer simmering stir chop chopped slice sliced dice diced oven pan pans
pot pots skillet stove kitchen onion onions garlic tomato tomatoes sauce salt pepper oil
butter flour sugar egg eggs rice pasta noodle noodles vegetable vegetables meat chicken
beef pork fish soup stew dish dishes meal meals dinner lunch breakfast taste tasty flavour
flavor season seasoning spice spices herb herbs basil cheese bread dough cream milk
ingredient ingredients food foods delicious grill grilled saute knife bowl plate dessert
cake bakery chef chefs cuisine potato potatoes carrot carrots lemon vinegar juice sweet
salty spicy crispy fresh mix mixture whisk peel boiled steam steamed snack sandwich salad
""".split())))

_WORD = re.compile(r"[a-zA-Z]+")


def keyword_score(text: str, keywords) -> dict:
    words = [w.lower() for w in _WORD.findall(text)]
    kw = set(keywords)
    hits = [w for w in words if w in kw or (w.endswith("s") and w[:-1] in kw)]
    return {"keyword_frac": len(hits) / len(words) if words else 0.0, "keyword_hits": hits}


@torch.no_grad()
def embedding_topic_distance(text: str, reference_texts, embedder_name: str) -> float:
    emb = load_embedder(embedder_name)
    refs = emb.encode(list(reference_texts), convert_to_tensor=True, normalize_embeddings=True)
    c = refs.mean(0)
    c = c / c.norm()
    e = emb.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
    return 1.0 - float(e @ c.to(e.device))


def topical_scores(text: str, reference_texts, embedder_name: str, keywords) -> dict:
    return {"minilm_cos_dist_to_topic": embedding_topic_distance(text, reference_texts, embedder_name),
            **keyword_score(text, keywords)}
