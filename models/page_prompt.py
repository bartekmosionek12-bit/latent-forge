"""Chat prompt for the product-page (indirect injection) scenario, split around the
attack slot so that a suffix can be optimised between a fixed prefix and a fixed tail."""
from __future__ import annotations

from evaluation.topic_data import PAGE_USER_TEMPLATE

_MARK = "\x00ATTACK\x00"
#: chat-template tokens after the user turn on Qwen2.5 (Ben-Tov / ActGCG target positions)
QWEN_ASSISTANT_HEADER = "<|im_end|>\n<|im_start|>assistant\n"


def user_message(page: str, attack: str, question: str) -> str:
    return PAGE_USER_TEMPLATE.format(page=page, attack=attack, question=question)


def page_prompt_parts(tokenizer, page: str, question: str) -> tuple[str, str]:
    """(before, after) strings of the fully rendered chat prompt; the attack text goes
    between them. ``before`` ends with the page text, ``after`` starts with the newline
    that closes the page and ends with the assistant header (generation starts there)."""
    full = tokenizer.apply_chat_template([{"role": "user", "content": user_message(page, _MARK, question)}],
                                         tokenize=False, add_generation_prompt=True)
    before, after = full.split(_MARK)
    return before, after


def n_template_tokens(tokenizer, after_ids) -> int:
    """Number of trailing chat-template tokens after the user's text (assistant header)."""
    hdr = tokenizer(QWEN_ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    tail = list(after_ids[-len(hdr):])
    if [int(x) for x in tail] != hdr:
        raise ValueError("prompt does not end with the Qwen assistant header; "
                         "template positions are only defined for Qwen2.5 chat models here")
    return len(hdr)
