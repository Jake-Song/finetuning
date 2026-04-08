"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import re
import json
from datasets import load_dataset
from tasks.common import Task

# -----------------------------
# Constraint checkers
# -----------------------------
def _compare(value: int, relation: str, target: int) -> bool:
    if relation == "at least":
        return value >= target
    if relation == "at most":
        return value <= target
    if relation == "less than":
        return value < target
    if relation == "more than":
        return value > target
    if relation == "exactly":
        return value == target
    return True


def check_no_comma(text: str, kw: dict) -> bool:
    return "," not in text


def check_number_words(text: str, kw: dict) -> bool:
    relation = kw.get("relation")
    num_words = kw.get("num_words")
    if relation is None or num_words is None:
        return True
    return _compare(len(text.split()), relation, num_words)


def check_number_sentences(text: str, kw: dict) -> bool:
    relation = kw.get("relation")
    num_sentences = kw.get("num_sentences")
    if relation is None or num_sentences is None:
        return True
    count = len(re.findall(r"[.!?]+(?:\s|$)", text))
    return _compare(count, relation, num_sentences)


def check_number_paragraphs(text: str, kw: dict) -> bool:
    num_paragraphs = kw.get("num_paragraphs")
    if num_paragraphs is None:
        return True
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return len(paragraphs) == num_paragraphs


def check_keywords_existence(text: str, kw: dict) -> bool:
    keywords = kw.get("keywords")
    if not keywords:
        return True
    text_lower = text.lower()
    return all(k.lower() in text_lower for k in keywords)


def check_forbidden_words(text: str, kw: dict) -> bool:
    forbidden = kw.get("forbidden_words")
    if not forbidden:
        return True
    text_lower = text.lower()
    return all(w.lower() not in text_lower for w in forbidden)


def check_keyword_frequency(text: str, kw: dict) -> bool:
    keyword = kw.get("keyword")
    frequency = kw.get("frequency")
    relation = kw.get("relation")
    if keyword is None or frequency is None or relation is None:
        return True
    count = text.lower().count(keyword.lower())
    return _compare(count, relation, frequency)


def check_english_lowercase(text: str, kw: dict) -> bool:
    return text == text.lower()


def check_english_capital(text: str, kw: dict) -> bool:
    return text == text.upper()


def check_title(text: str, kw: dict) -> bool:
    return bool(re.match(r"^#\s", text.strip()))


def check_number_highlighted_sections(text: str, kw: dict) -> bool:
    num_highlights = kw.get("num_highlights")
    if num_highlights is None:
        return True
    count = len(re.findall(r"(?m)^\*{2,}[^*]+\*{2,}", text))
    if count == 0:
        count = len(re.findall(r"(?m)^#{1,6}\s", text))
    return _compare(count, "at least", num_highlights)


def check_number_bullet_lists(text: str, kw: dict) -> bool:
    num_bullets = kw.get("num_bullets")
    if num_bullets is None:
        return True
    count = len(re.findall(r"(?m)^[\*\-]\s", text))
    return _compare(count, "at least", num_bullets)


def check_json_format(text: str, kw: dict) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_postscript(text: str, kw: dict) -> bool:
    marker = kw.get("postscript_marker", "P.S.")
    if marker is None:
        marker = "P.S."
    return marker in text


def check_end_checker(text: str, kw: dict) -> bool:
    end_phrase = kw.get("end_phrase")
    if end_phrase is None:
        return True
    return text.strip().endswith(end_phrase)


def check_quotation(text: str, kw: dict) -> bool:
    stripped = text.strip()
    return (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("\u201c") and stripped.endswith("\u201d")
    )


def check_repeat_prompt(text: str, kw: dict) -> bool:
    prompt_to_repeat = kw.get("prompt_to_repeat")
    if prompt_to_repeat is None:
        return True
    return prompt_to_repeat in text


CHECKERS = {
    "punctuation:no_comma": check_no_comma,
    "length_constraints:number_words": check_number_words,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "keywords:existence": check_keywords_existence,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:frequency": check_keyword_frequency,
    "change_case:english_lowercase": check_english_lowercase,
    "change_case:english_capital": check_english_capital,
    "detectable_format:title": check_title,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:json_format": check_json_format,
    "detectable_content:postscript": check_postscript,
    "startend:end_checker": check_end_checker,
    "startend:quotation": check_quotation,
    "combination:repeat_prompt": check_repeat_prompt,
}


class IFRL(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in "IF-RL", "IF-RL subset must be IF-RL"
        self.ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", subset, split="train").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        prompt = row['prompt'] # string of the question prompt
        instruction_id_list = row['instruction_id_list'] # list of instruction ids
        kwargs = row['kwargs'] # dictionary of kwargs

        return prompt, instruction_id_list, kwargs


    def reward(self, completions, instruction_id_list, kwargs, **_ignored) -> list[float]:
        rewards = []
        for i, completion in enumerate(completions):
            # extract text from conversational format
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            else:
                text = str(completion)
            text = text.strip()

            if not text:
                rewards.append(0.0)
                continue

            ids = instruction_id_list[i]
            kws = kwargs[i]
            if not ids:
                rewards.append(0.5)
                continue

            passed = 0
            for constraint_id, kw in zip(ids, kws):
                checker = CHECKERS.get(constraint_id)
                if checker is None:
                    passed += 1  # skip unknown constraints
                    continue
                if checker(text, kw):
                    passed += 1

            rewards.append(passed / len(ids))
        return rewards
   
if __name__ == "__main__":
    task = IFRL(subset="IF-RL", split="train")
    ex = task.get_example(0)
    print(ex)