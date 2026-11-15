"""Semantic (not list-based) token blocking for the lexical-vs-latent test.

A token is "topic-specific" if a separate encoder (default all-MiniLM-L6-v2) places
the token's text closer to the topic centroid than to the centroid of EVERY control
topic, and above the vocabulary median similarity:

    spec(t) = sim(t, topic) - max_k sim(t, control_k)       block if spec > margin
                                                             and sim(t, topic) > median

Comparing against control topics is what keeps function words ("the", "and") out:
they are about equally close to every topic. The model's own input-embedding matrix
was rejected as the similarity space: in distilgpt2 its nearest tokens to the topic
are function words (frequency, not meaning)."""
from __future__ import annotations

import torch

from models import load_embedder

CONTROL_TOPICS = {
    "politics": ["The city council voted on the new zoning ordinance after a long debate.",
                 "The mayor announced a budget proposal for public transport."],
    "sports": ["The striker scored twice in the second half to win the match.",
               "The team trained hard before the championship final."],
    "computing": ["The server crashed after the software update corrupted the database.",
                  "Install the package and restart the terminal to apply the changes."],
    "weather": ["Heavy rain and strong winds are expected across the region tomorrow.",
                "The forecast predicts sunny skies and mild temperatures this weekend."],
    "finance": ["The bank raised interest rates as inflation continued to climb.",
                "Investors sold shares after the quarterly earnings report."],
    "music": ["The band played their new album to a sold-out concert hall.",
              "She practised the piano sonata every evening for a month."],
    "travel": ["We booked a hotel near the station and took the train to the coast.",
               "The flight was delayed, so we explored the airport for hours."],
}


@torch.no_grad()
def topic_specificity(tokenizer, topic_texts, control_topics=None,
                      embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Returns (sim_topic [V], spec [V]) for every token id of ``tokenizer``."""
    control_topics = control_topics or CONTROL_TOPICS
    emb = load_embedder(embedder_name)
    strs = [tokenizer.decode([i]).strip() for i in range(len(tokenizer))]
    E = emb.encode(strs, batch_size=1024, convert_to_tensor=True, normalize_embeddings=True).cpu()

    def centroid(texts):
        c = emb.encode(list(texts), convert_to_tensor=True, normalize_embeddings=True).mean(0).cpu()
        return c / c.norm()

    sim = E @ centroid(topic_texts)
    others = torch.stack([E @ centroid(v) for v in control_topics.values()], dim=1)
    return sim, sim - others.max(dim=1).values


def semantic_block_ids(tokenizer, topic_texts, margin: float = 0.0, **kw) -> tuple[set[int], dict]:
    sim, spec = topic_specificity(tokenizer, topic_texts, **kw)
    mask = (spec > margin) & (sim > sim.median())
    ids = set(mask.nonzero()[:, 0].tolist())
    return ids, {"sim": sim, "spec": spec, "n_blocked": len(ids), "median_sim": sim.median().item()}
