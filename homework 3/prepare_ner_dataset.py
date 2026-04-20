from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

BOOK_PATH = Path(__file__).with_name("book.txt")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("dataset")

VOCAB_NER_TAGS = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-RACE",
    6: "I-RACE",
}

TOKEN_RE = re.compile(
    r"[A-Za-z\u0400-\u04FF]+(?:[’'`-][A-Za-z\u0400-\u04FF]+)*|\d+|[«»()\[\]{}.,!?;:…—–-]"
)
WORD_RE = re.compile(r"[A-Za-z\u0400-\u04FF]+(?:[’'`-][A-Za-z\u0400-\u04FF]+)*|\d+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[«–—-]|[A-ZА-ЯЁ])")

SKIP_DRAMATIS_HEADERS = {
    "Действующие лица",
    "Тисте эдур",
    "Рабы-летерийцы у тисте эдур",
    "Летери во дворце",
    "На севере",
    "В городе Летерас",
    "Другие",
    "Та, кто внутри",
}

SKIP_PARAGRAPH_PREFIXES = (
    "Пролог",
    "Глава ",
    "Книга ",
    "Видение ",
    "Песнь ",
    "Песня ",
    "Из ",
    "Год ",
)

ADJECTIVE_LIKE_ENDINGS = (
    "ый",
    "ий",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
)

PERSON_EXTRA = [
    "Маэль",
    "Матерь Тьма",
    "Мать Тьма",
    "Отец Тень",
    "Королева Сновидений",
    "Странник",
    "К’рул",
    "Драконус",
    "Оссерк",
    "Кильмандарос",
    "Сечул Лат",
    "Сестра Холодных ночей",
    "Менандор",
    "Ханради Халаг",
    "Ханради Кхалаг",
]

PERSON_ALIASES = {
    "Йан Товис": ["Сумрак"],
    "Пернатая Ведьма": ["Ведьма", "Пернатая"],
    "Скабандари Кровавый глаз": ["Кровавый глаз"],
    "Матерь Тьма": ["Тьма"],
    "Мать Тьма": ["Тьма"],
    "Отец Тень": ["Тень"],
}

LOCATIONS = [
    "Эмурланн",
    "Куральд Эмурланн",
    "Куральд Галейн",
    "Галейн",
    "Старвальд Демелейн",
    "Демелейн",
    "Омтоз Феллак",
    "Морн",
    "Летер",
    "Летерас",
    "Предел фентов",
    "Пустая Обитель",
    "Немил",
    "Трейт",
    "Калач",
    "Лежбище Калача",
    "Обитель Азатов",
    "Обитель Льда",
    "Вечный дом",
    "Дом Мертвых",
]

RACES = [
    "Старшие боги",
    "тисте эдур",
    "эдур",
    "тисте анди",
    "анди",
    "к’чейн че’маллей",
    "к'чейн че'маллей",
    "к’елль",
    "к'елль",
    "короткохвостые",
    "короткохвостый",
    "яггут",
    "яггуты",
    "имасс",
    "имассы",
    "форкрул ассейл",
    "форкрул ассейлы",
    "летериец",
    "летерийцы",
    "летери",
    "нерек",
    "нереки",
    "фараэд",
    "фараэды",
    "тартенал",
    "тартеналы",
    "мекрос",
    "мекросы",
    "нахт",
    "нахты",
    "элейнт",
    "бхока’рал",
    "бхока’ралы",
]

LABEL_START = {"PER": 1, "LOC": 3, "RACE": 5}


def normalize_text(text: str) -> str:
    replacements = {
        "\u2014": "—",
        "\u2015": "—",
        "\u2212": "-",
        "\u02bc": "’",
        "\u2018": "’",
        "\u2019": "’",
        "`": "’",
        "´": "’",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_token(token: str) -> str:
    return normalize_text(token).lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def is_word(token: str) -> bool:
    return bool(WORD_RE.fullmatch(token))


def strip_parenthetical_aliases(name: str) -> tuple[str, list[str]]:
    aliases: list[str] = []
    if "（" in name or "）" in name:
        name = name.replace("（", "(").replace("）", ")")

    match = re.search(r"\(([^)]*)\)", name)
    if match:
        alias = match.group(1).strip().strip("«»\"'")
        if alias:
            aliases.append(alias)
        name = (name[: match.start()] + name[match.end() :]).strip()

    return name.strip(), aliases


def parse_characters(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        start = lines.index("Действующие лица") + 1
        end = lines.index("Пролог")
    except ValueError as exc:
        raise RuntimeError("Не удалось найти блок 'Действующие лица' или 'Пролог'.") from exc

    people: list[str] = []
    for raw_line in lines[start:end]:
        line = raw_line.strip()
        if not line or line in SKIP_DRAMATIS_HEADERS:
            continue
        candidate = line.split(" , ", 1)[0].strip()
        if candidate in SKIP_DRAMATIS_HEADERS:
            continue
        candidate, aliases = strip_parenthetical_aliases(candidate)
        if candidate and candidate not in people:
            people.append(candidate)
        for alias in aliases:
            if alias and alias not in people:
                people.append(alias)
    return people


def is_heading_like(paragraph: str) -> bool:
    clean = paragraph.strip()
    if not clean:
        return True
    for prefix in SKIP_PARAGRAPH_PREFIXES:
        if clean.startswith(prefix):
            return True
    if clean.startswith("Восхождение "):
        return True
    if clean.startswith("Седьмого завершения "):
        return True
    return False


def collect_story_units(text: str) -> list[str]:
    lines = text.splitlines()
    try:
        prologue_index = lines.index("Пролог")
    except ValueError as exc:
        raise RuntimeError("Не удалось найти 'Пролог'.") from exc

    units: list[str] = []
    i = prologue_index + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if any(ch in line for ch in ".?!…"):
            break
        if line != "Пролог":
            units.append(line)
        i += 1

    remaining = "\n".join(lines[i:])
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(remaining) if p.strip()]
    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        if is_heading_like(paragraph):
            continue
        if not any(ch in paragraph for ch in ".?!…"):
            continue
        parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(paragraph) if part.strip()]
        units.extend(parts)

    return units


def adjective_like(token: str) -> bool:
    low = normalize_token(token)
    return any(low.endswith(ending) for ending in ADJECTIVE_LIKE_ENDINGS)


def inflect_token(token: str) -> set[str]:
    low = normalize_token(token)
    forms = {low}

    if not is_word(token):
        return forms

    if low.endswith(("ый", "ой")):
        stem = low[:-2]
        forms.update(
            {
                f"{stem}ого",
                f"{stem}ому",
                f"{stem}ым",
                f"{stem}ом",
                f"{stem}ая",
                f"{stem}ые",
                f"{stem}ых",
                f"{stem}ыми",
            }
        )
        return forms

    if low.endswith("ий"):
        stem = low[:-2]
        forms.update(
            {
                f"{stem}его",
                f"{stem}ему",
                f"{stem}им",
                f"{stem}ем",
                f"{stem}яя",
                f"{stem}ие",
                f"{stem}их",
                f"{stem}ими",
            }
        )
        return forms

    if low.endswith("ая"):
        stem = low[:-2]
        forms.update(
            {
                f"{stem}ой",
                f"{stem}ую",
                f"{stem}ые",
                f"{stem}ых",
                f"{stem}ыми",
                f"{stem}ого",
                f"{stem}ому",
                f"{stem}ым",
                f"{stem}ом",
            }
        )
        return forms

    if low.endswith("яя"):
        stem = low[:-2]
        forms.update(
            {
                f"{stem}ей",
                f"{stem}юю",
                f"{stem}ие",
                f"{stem}их",
                f"{stem}ими",
                f"{stem}его",
                f"{stem}ему",
                f"{stem}им",
                f"{stem}ем",
            }
        )
        return forms

    if low.endswith("ые"):
        stem = low[:-2]
        forms.update(
            {
                f"{stem}ых",
                f"{stem}ыми",
                f"{stem}ый",
                f"{stem}ого",
                f"{stem}ому",
                f"{stem}ым",
                f"{stem}ом",
                f"{stem}ая",
                f"{stem}ую",
                f"{stem}ое",
            }
        )
        return forms

    if low.endswith("ие"):
        stem = low[:-2]
        forms.update(
            {
                f"{stem}их",
                f"{stem}ими",
                f"{stem}ий",
                f"{stem}его",
                f"{stem}ему",
                f"{stem}им",
                f"{stem}ем",
                f"{stem}яя",
                f"{stem}юю",
                f"{stem}ее",
            }
        )
        return forms

    if low.endswith("а"):
        stem = low[:-1]
        forms.update({f"{stem}у", f"{stem}е", f"{stem}ой", f"{stem}ою", f"{stem}ам", f"{stem}ами", f"{stem}ах"})
        if stem.endswith(("г", "к", "х", "ж", "ч", "ш", "щ", "ц")):
            forms.add(f"{stem}и")
        else:
            forms.add(f"{stem}ы")
        return forms

    if low.endswith("я"):
        stem = low[:-1]
        forms.update({f"{stem}ю", f"{stem}е", f"{stem}ей", f"{stem}ею", f"{stem}ям", f"{stem}ями", f"{stem}ях", f"{stem}и"})
        return forms

    if low.endswith(("ь", "й")):
        stem = low[:-1]
        forms.update({f"{stem}я", f"{stem}ю", f"{stem}ем", f"{stem}е", f"{stem}и", f"{stem}ям", f"{stem}ями", f"{stem}ях"})
        return forms

    if low.endswith(("ы", "и")):
        stem = low[:-1]
        forms.update({f"{stem}ов", f"{stem}ам", f"{stem}ами", f"{stem}ах"})
        return forms

    if low.endswith(("о", "е", "у", "ю")):
        return forms

    forms.update({f"{low}а", f"{low}у", f"{low}ом", f"{low}е", f"{low}ы", f"{low}ов", f"{low}ам", f"{low}ами", f"{low}ах"})
    return forms


def build_surface_forms(entity: str) -> set[tuple[str, ...]]:
    words = [word for word in tokenize(entity) if is_word(word)]
    if not words:
        return set()
    token_forms = [sorted(inflect_token(word)) for word in words]
    return {tuple(combo) for combo in itertools.product(*token_forms)}


def build_person_entities(text: str) -> set[str]:
    people = parse_characters(text)
    people.extend(name for name in PERSON_EXTRA if name not in people)

    first_token_counts = Counter()
    last_token_counts = Counter()
    split_names: dict[str, list[str]] = {}
    for name in people:
        tokens = [token for token in tokenize(name) if is_word(token)]
        if not tokens:
            continue
        split_names[name] = tokens
        first_token_counts[normalize_token(tokens[0])] += 1
        last_token_counts[normalize_token(tokens[-1])] += 1

    entities: set[str] = set(people)
    for name, tokens in split_names.items():
        if len(tokens) == 1:
            continue

        first = tokens[0]
        if first_token_counts[normalize_token(first)] == 1 and not adjective_like(first):
            entities.add(first)

        last = tokens[-1]
        if last_token_counts[normalize_token(last)] == 1 and not adjective_like(last):
            entities.add(last)

    for base, aliases in PERSON_ALIASES.items():
        if base in entities or base in people:
            entities.update(aliases)

    return entities


def build_patterns(text: str) -> dict[str, list[tuple[tuple[str, ...], str]]]:
    per_entities = build_person_entities(text)
    loc_entities = set(LOCATIONS)
    race_entities = set(RACES)

    patterns: list[tuple[tuple[str, ...], str]] = []
    for entity in sorted(per_entities):
        for form in build_surface_forms(entity):
            patterns.append((form, "PER"))
    for entity in sorted(loc_entities):
        for form in build_surface_forms(entity):
            patterns.append((form, "LOC"))
    for entity in sorted(race_entities):
        for form in build_surface_forms(entity):
            patterns.append((form, "RACE"))

    patterns.sort(key=lambda item: (-len(item[0]), item[1], item[0]))

    by_first: dict[str, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    for form, label in patterns:
        by_first[form[0]].append((form, label))
    return by_first


def tag_tokens(tokens: list[str], patterns_by_first: dict[str, list[tuple[tuple[str, ...], str]]]) -> list[int]:
    normalized = [normalize_token(token) if is_word(token) else token for token in tokens]
    tags = [0] * len(tokens)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not is_word(token):
            i += 1
            continue

        candidates = patterns_by_first.get(normalized[i], [])
        matched = None
        for form, label in candidates:
            if label in {"PER", "LOC"} and not tokens[i][0].isupper():
                continue
            span = len(form)
            if tuple(normalized[i : i + span]) == form:
                matched = (span, label)
                break

        if matched is None:
            i += 1
            continue

        span, label = matched
        start_tag = LABEL_START[label]
        tags[i] = start_tag
        for j in range(1, span):
            if i + j < len(tags):
                tags[i + j] = start_tag + 1
        i += span

    return tags


def make_samples(units: list[str], patterns_by_first: dict[str, list[tuple[tuple[str, ...], str]]]) -> list[dict[str, list]]:
    samples: list[dict[str, list]] = []
    for unit in units:
        tokens = tokenize(unit)
        tags = tag_tokens(tokens, patterns_by_first)
        samples.append({"tokens": tokens, "ner_tags": tags})
    return samples


def format_dataset(samples: list[dict[str, list]]) -> str:
    lines = [
        "{",
        '    "vocab_ner_tags": {',
        '        "0": "O",',
        '        "1": "B-PER",',
        '        "2": "I-PER",',
        '        "3": "B-LOC",',
        '        "4": "I-LOC",',
        '        "5": "B-RACE",',
        '        "6": "I-RACE"',
        "    },",
        '    "sentencies": [',
    ]

    for idx, sample in enumerate(samples):
        suffix = "," if idx < len(samples) - 1 else ""
        lines.extend(
            [
                "        {",
                f"            'tokens': {repr(sample['tokens'])},",
                f"            'ner_tags': {sample['ner_tags']}",
                f"        }}{suffix}",
            ]
        )

    lines.extend(["    ]", "}"])
    return "\n".join(lines) + "\n"


def write_split(path: Path, samples: list[dict[str, list]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_dataset(samples), encoding="utf-8")


def build_unknown_report(samples: list[dict[str, list]]) -> str:
    counter = Counter()
    for sample in samples:
        tokens = sample["tokens"]
        tags = sample["ner_tags"]
        for token, tag in zip(tokens, tags):
            if tag != 0 or not is_word(token):
                continue
            if token[0].isupper():
                counter[token] += 1
    lines = ["Top unmatched title-cased tokens:"]
    for token, count in counter.most_common(80):
        lines.append(f"{count:>3} {token}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка NER-датасета по книге.")
    parser.add_argument("--train", type=int, default=1000, help="Число предложений для train.")
    parser.add_argument("--val", type=int, default=200, help="Число предложений для val.")
    parser.add_argument("--test", type=int, default=200, help="Число предложений для test.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Каталог, куда будут записаны train.txt, val.txt и test.txt.",
    )
    parser.add_argument("--report", action="store_true", help="Показать краткий отчёт по неразмеченным словам.")
    args = parser.parse_args()

    text = normalize_text(BOOK_PATH.read_text(encoding="utf-8"))
    units = collect_story_units(text)
    required = args.train + args.val + args.test
    if len(units) < required:
        raise RuntimeError(f"Недостаточно предложений: найдено {len(units)}, требуется {required}.")

    selected_units = units[:required]
    patterns_by_first = build_patterns(text)
    samples = make_samples(selected_units, patterns_by_first)
    output_dir = Path(args.output_dir)

    train_samples = samples[: args.train]
    val_samples = samples[args.train : args.train + args.val]
    test_samples = samples[args.train + args.val : required]

    write_split(output_dir / "train.txt", train_samples)
    write_split(output_dir / "val.txt", val_samples)
    write_split(output_dir / "test.txt", test_samples)

    print(f"Всего смысловых единиц доступно: {len(units)}")
    print(f"Сохранено: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    print(f"Файлы записаны в: {output_dir}")

    if args.report:
        print()
        print(build_unknown_report(samples))


if __name__ == "__main__":
    main()
