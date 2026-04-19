import re
import json
import string
import collections
from typing import Any

CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

PUBLIC_IFEVAL_INSTRUCTION_IDS = frozenset({
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "startend:quotation",
})

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


def _normalize_kwargs(kw: dict | None) -> dict:
    return kw if isinstance(kw, dict) else {}


def _count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def _count_sentences(text: str) -> int:
    return len(re.findall(r"[.!?]+(?:\s|$)", text))


def _split_star_paragraphs(text: str) -> list[str] | None:
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    valid = []
    for idx, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            valid.append(paragraph)
            continue
        if idx == 0 or idx == len(paragraphs) - 1:
            continue
        return None
    return valid


def _split_double_newline_paragraphs(text: str) -> list[str]:
    return [paragraph for paragraph in re.split(r"\n\n", text) if paragraph.strip()]


def _detect_language(text: str) -> str | None:
    try:
        from langdetect import LangDetectException, detect

        try:
            return detect(text)
        except LangDetectException:
            return None
    except ImportError:
        # Fallback only supports coarse script detection when `langdetect`
        # is unavailable locally. Keep English working for dry-runs/tests.
        if re.search(r"[\u0400-\u04FF]", text):
            return "ru"
        if re.search(r"[\u4E00-\u9FFF]", text):
            return "zh"
        if re.search(r"[A-Za-z]", text):
            return "en"
        return None


def check_no_comma(text: str, kw: dict) -> bool:
    return "," not in text


def check_number_words(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    relation = kw.get("relation")
    num_words = kw.get("num_words")
    if relation is None or num_words is None:
        return True
    return _compare(_count_words(text), relation, num_words)


def check_number_sentences(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    relation = kw.get("relation")
    num_sentences = kw.get("num_sentences")
    if relation is None or num_sentences is None:
        return True
    return _compare(_count_sentences(text), relation, num_sentences)


def check_number_paragraphs(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_paragraphs = kw.get("num_paragraphs")
    if num_paragraphs is None:
        return True
    paragraphs = _split_star_paragraphs(text)
    return paragraphs is not None and len(paragraphs) == num_paragraphs


def check_keywords_existence(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keywords = kw.get("keywords")
    if not keywords:
        return True
    return all(re.search(re.escape(keyword), text, flags=re.IGNORECASE) for keyword in keywords)


def check_forbidden_words(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    forbidden = kw.get("forbidden_words")
    if not forbidden:
        return True
    return all(
        not re.search(r"\b" + re.escape(word) + r"\b", text, flags=re.IGNORECASE)
        for word in forbidden
    )


def check_keyword_frequency(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    keyword = kw.get("keyword")
    frequency = kw.get("frequency")
    relation = kw.get("relation")
    if keyword is None or frequency is None or relation is None:
        return True
    count = len(re.findall(re.escape(keyword), text, flags=re.IGNORECASE))
    return _compare(count, relation, frequency)


def check_english_lowercase(text: str, kw: dict) -> bool:
    detected = _detect_language(text)
    return text.islower() and detected == "en"


def check_english_capital(text: str, kw: dict) -> bool:
    detected = _detect_language(text)
    return text.isupper() and detected == "en"


def check_title(text: str, kw: dict) -> bool:
    return any(title.lstrip("<").rstrip(">").strip() for title in re.findall(r"<<[^\n]+>>", text))


def check_number_highlighted_sections(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_highlights = kw.get("num_highlights")
    if num_highlights is None:
        return True
    count = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", text):
        if highlight.strip("*").strip():
            count += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", text):
        if highlight.removeprefix("**").removesuffix("**").strip():
            count += 1
    return count >= num_highlights


def check_number_bullet_lists(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_bullets = kw.get("num_bullets")
    if num_bullets is None:
        return True
    count = len(re.findall(r"^\s*\*[^\*].*$", text, flags=re.MULTILINE))
    count += len(re.findall(r"^\s*-.*$", text, flags=re.MULTILINE))
    return count == num_bullets


def check_json_format(text: str, kw: dict) -> bool:
    candidate = (
        text.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(candidate)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_postscript(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    marker = kw.get("postscript_marker", "P.S.")
    if marker is None:
        marker = "P.S."
    value = text.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def check_end_checker(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    end_phrase = kw.get("end_phrase")
    if end_phrase is None:
        return True
    return text.strip().strip('"').lower().endswith(str(end_phrase).strip().lower())


def check_quotation(text: str, kw: dict) -> bool:
    stripped = text.strip()
    return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'


def check_repeat_prompt(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    prompt_to_repeat = kw.get("prompt_to_repeat")
    if prompt_to_repeat is None:
        return True
    return text.strip().lower().startswith(str(prompt_to_repeat).strip().lower())


def check_keyword_letter_frequency(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    letter = kw.get("letter")
    let_frequency = kw.get("let_frequency")
    let_relation = kw.get("let_relation")
    if not letter or let_frequency is None or let_relation is None:
        return True
    normalized_letter = str(letter).strip().lower()
    if len(normalized_letter) != 1 or normalized_letter not in string.ascii_lowercase:
        return False
    count = collections.Counter(text.lower())[normalized_letter]
    return _compare(count, let_relation, let_frequency)


def check_response_language(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    language = kw.get("language")
    if not language:
        return True
    detected = _detect_language(text)
    if detected is None:
        return False
    expected = str(language).lower()
    detected = detected.lower()
    return detected == expected or detected.split("-")[0] == expected.split("-")[0]


def check_nth_paragraph_first_word(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_paragraphs = kw.get("num_paragraphs")
    nth_paragraph = kw.get("nth_paragraph")
    first_word = kw.get("first_word")
    if num_paragraphs is None or nth_paragraph is None or first_word is None:
        return True

    paragraphs = _split_double_newline_paragraphs(text)
    if len(paragraphs) != num_paragraphs or nth_paragraph <= 0 or nth_paragraph > len(paragraphs):
        return False

    paragraph = paragraphs[nth_paragraph - 1].strip()
    if not paragraph:
        return False

    word = paragraph.split()[0].lstrip("'\"")
    extracted = []
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        extracted.append(letter.lower())
    return "".join(extracted) == str(first_word).lower()


def check_number_placeholders(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    num_placeholders = kw.get("num_placeholders")
    if num_placeholders is None:
        return True
    return len(re.findall(r"\[.*?\]", text)) >= num_placeholders


def check_constrained_response(text: str, kw: dict) -> bool:
    value = text.strip()
    return any(option in value for option in CONSTRAINED_RESPONSE_OPTIONS)


def check_multiple_sections(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    section_spliter = kw.get("section_spliter")
    num_sections = kw.get("num_sections")
    if not section_spliter or num_sections is None:
        return True
    pattern = r"\s?" + re.escape(str(section_spliter).strip()) + r"\s?\d+\s?"
    return max(len(re.split(pattern, text)) - 1, 0) >= num_sections


def check_two_responses(text: str, kw: dict) -> bool:
    responses = text.split("******")
    valid_responses = []
    for index, response in enumerate(responses):
        if not response.strip():
            if index != 0 and index != len(responses) - 1:
                return False
            continue
        valid_responses.append(response)
    return len(valid_responses) == 2 and valid_responses[0].strip() != valid_responses[1].strip()


def check_capital_word_frequency(text: str, kw: dict) -> bool:
    kw = _normalize_kwargs(kw)
    frequency = kw.get("capital_frequency")
    relation = kw.get("capital_relation")
    if frequency is None or relation is None:
        return True
    words = re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b", text)
    count = sum(1 for word in words if word.isupper())
    return _compare(count, relation, frequency)


CHECKERS = {
    "punctuation:no_comma": check_no_comma,
    "length_constraints:number_words": check_number_words,
    "length_constraints:number_sentences": check_number_sentences,
    "length_constraints:number_paragraphs": check_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": check_nth_paragraph_first_word,
    "keywords:existence": check_keywords_existence,
    "keywords:forbidden_words": check_forbidden_words,
    "keywords:frequency": check_keyword_frequency,
    "keywords:letter_frequency": check_keyword_letter_frequency,
    "language:response_language": check_response_language,
    "change_case:english_lowercase": check_english_lowercase,
    "change_case:english_capital": check_english_capital,
    "change_case:capital_word_frequency": check_capital_word_frequency,
    "detectable_format:title": check_title,
    "detectable_format:number_highlighted_sections": check_number_highlighted_sections,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:constrained_response": check_constrained_response,
    "detectable_format:multiple_sections": check_multiple_sections,
    "detectable_format:json_format": check_json_format,
    "detectable_content:number_placeholders": check_number_placeholders,
    "detectable_content:postscript": check_postscript,
    "startend:end_checker": check_end_checker,
    "startend:quotation": check_quotation,
    "combination:repeat_prompt": check_repeat_prompt,
    "combination:two_responses": check_two_responses,
}


def summarize_constraint_evaluation(
    text: str,
    instruction_ids: list[str],
    kwargs_list: list[dict],
) -> dict[str, Any]:
    stripped = text.strip()
    results: dict[str, bool | None] = {}
    passed = 0
    implemented = 0

    for idx, constraint_id in enumerate(instruction_ids):
        kw = _normalize_kwargs(kwargs_list[idx] if idx < len(kwargs_list) else None)
        checker = CHECKERS.get(constraint_id)
        if checker is None:
            results[constraint_id] = None
            continue
        implemented += 1
        result = checker(stripped, kw)
        results[constraint_id] = result
        if result:
            passed += 1

    if not stripped:
        reward = 0.0
    elif not instruction_ids:
        reward = 0.5
    elif implemented == 0:
        reward = 0.0
    else:
        reward = passed / implemented

    all_passed = (
        bool(stripped)
        and bool(instruction_ids)
        and implemented == len(instruction_ids)
        and passed == implemented
    )
    return {
        "results": results,
        "reward": reward,
        "implemented": implemented,
        "passed": passed,
        "all_passed": all_passed,
    }


def evaluate_constraints(text: str, instruction_ids: list[str], kwargs_list: list[dict]) -> dict[str, bool | None]:
    return summarize_constraint_evaluation(text, instruction_ids, kwargs_list)["results"]

def compute_rewards(
    completions: list[str],
    instruction_id_list: list[list[str]],
    kwargs_list: list[list[dict]],
) -> list[float]:
    rewards = []
    for text, ids, kws in zip(completions, instruction_id_list, kwargs_list):
        rewards.append(summarize_constraint_evaluation(text, ids, kws)["reward"])
    return rewards

if __name__ == "__main__":
    completions = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
    instruction_id_list = [["keywords:existence"], ["keywords:existence"], ["keywords:existence"]]
    kwargs_list = [{"keywords": ["yes", "no", "maybe"]}, {"keywords": ["yes", "no", "maybe"]}, {"keywords": ["yes", "no", "maybe"]}]
    rewards = compute_rewards(completions, instruction_id_list, kwargs_list)
    print(rewards)