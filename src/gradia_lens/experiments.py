"""Versioned, hand-written contrast fitting and separately partitioned test cases.

These tests measure next-token lexical preferences, not truthfulness or safety.
Confirmation examples are excluded from adaptive search acquisition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    desired: tuple[str, ...]
    undesired: tuple[str, ...]
    split: str
    partition: str
    label: str


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    description: str
    training_pairs: tuple[tuple[str, str], ...]
    cases: tuple[Case, ...]
    default_prompt: str
    default_layer: int = 10
    default_strength: float = 0.35

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def catalog_item(self) -> dict:
        target = next(case for case in self.cases if case.split == "target")
        counts = {
            split: sum(c.split == split for c in self.cases) for split in ("target", "near", "unrelated")
        }
        counts["training"] = len(self.training_pairs)
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "default_prompt": self.default_prompt,
            "default_layer": self.default_layer,
            "default_strength": self.default_strength,
            "target_tokens": list(target.desired),
            "contrast_tokens": list(target.undesired),
            "counts": counts,
            "dataset_sha256": self.digest(),
            "interpretation_limit": (
                "A contrast-trained residual direction and hand-written lexical probes; "
                "not a complete concept, a causal circuit, or a safety guarantee. "
                "Discovery cases guide search; confirmation cases audit chosen candidates."
            ),
        }


def _cases(prefix: str, targets, near, unrelated) -> tuple[Case, ...]:
    result: list[Case] = []
    for split, entries in (
        ("target", targets),
        ("near", near),
        ("unrelated", unrelated),
    ):
        for index, (prompt, desired, undesired, label) in enumerate(entries):
            # Last two cases in each family are never scored by search acquisition.
            partition = "discovery" if index < len(entries) - 2 else "confirmation"
            result.append(
                Case(
                    f"{prefix}-{split}-{index + 1}",
                    prompt,
                    tuple(desired),
                    tuple(undesired),
                    split,
                    partition,
                    label,
                )
            )
    return tuple(result)


POS = (" good", " great", " excellent", " wonderful")
NEG = (" bad", " awful", " terrible", " horrible")
SPACE = (" space", " moon", " planet", " stars")
OCEAN = (" sea", " ocean", " water", " beach")
FUTURE = (" will",)
PAST = (" was", " had", " did")


EXPERIMENTS = {
    "positive-tone": Experiment(
        id="positive-tone",
        title="Positive tone",
        description="Add a direction fitted from positive versus negative review contexts.",
        training_pairs=tuple(
            zip(  # noqa: B905 — optional Python 3.9 worker
                (
                    "The movie was delightful. Overall, it was",
                    "The meal tasted delicious. My experience was",
                    "The concert impressed every listener. It was",
                    "The hotel offered excellent service. The stay was",
                    "The book was engaging and inspiring. Reading it was",
                    "The game was exciting and fun. Playing it was",
                    "The show exceeded my expectations. I thought it was",
                    "The product worked beautifully. Its quality was",
                ),
                (
                    "The movie was dreadful. Overall, it was",
                    "The meal tasted disgusting. My experience was",
                    "The concert disappointed every listener. It was",
                    "The hotel offered terrible service. The stay was",
                    "The book was boring and depressing. Reading it was",
                    "The game was frustrating and tedious. Playing it was",
                    "The show fell short of my expectations. I thought it was",
                    "The product failed completely. Its quality was",
                ),
            )
        ),
        cases=_cases(
            "tone",
            [
                ("My experience at the cafe was", POS, NEG, "Unseen cafe review"),
                (
                    "After using the new app, I thought it was",
                    POS,
                    NEG,
                    "Unseen app review",
                ),
                (
                    "The audience described the performance as",
                    POS,
                    NEG,
                    "Unseen performance",
                ),
                (
                    "The museum tour left me thinking it was",
                    POS,
                    NEG,
                    "Sealed museum paraphrase",
                ),
                (
                    "I tried the workshop and found it",
                    POS,
                    NEG,
                    "Sealed workshop paraphrase",
                ),
            ],
            [
                (
                    "The food was rotten and made us sick. It tasted",
                    NEG,
                    POS,
                    "Explicitly negative food",
                ),
                (
                    "The service was rude and the room was filthy. The hotel was",
                    NEG,
                    POS,
                    "Explicitly negative hotel",
                ),
                (
                    "The instructions were misleading and the device broke. It was",
                    NEG,
                    POS,
                    "Explicitly negative device",
                ),
                (
                    "The plot was incoherent and the acting was wooden. The film was",
                    NEG,
                    POS,
                    "Sealed negative film",
                ),
            ],
            [
                (
                    "The capital of France is",
                    (" Paris",),
                    (" London", " Rome"),
                    "France capital",
                ),
                (
                    "Two plus two equals",
                    (" four", " 4"),
                    (" five", " 5"),
                    "Basic arithmetic",
                ),
                ("The opposite of hot is", (" cold",), (" warm",), "Sealed antonym"),
                (
                    "The largest planet in our solar system is",
                    (" Jupiter",),
                    (" Earth", " Mars"),
                    "Sealed planet fact",
                ),
            ],
        ),
        default_prompt="My experience at the cafe was",
    ),
    "space-association": Experiment(
        id="space-association",
        title="Space association",
        description="Promote astronomical associations using space versus ocean contrast contexts.",
        training_pairs=tuple(
            zip(  # noqa: B905 — optional Python 3.9 worker
                (
                    "The astronaut looked through the spaceship window. Outside was",
                    "The rocket passed the atmosphere. It entered",
                    "The astronomer aimed the telescope upward. She studied",
                    "The lunar mission carried explorers. Their destination was",
                    "The satellite orbited above Earth. Around it was",
                    "The science fiction crew traveled among distant worlds. They explored",
                    "The observatory searched the night sky. It observed",
                    "The spacecraft left our planet. It traveled through",
                ),
                (
                    "The diver looked through the submarine window. Outside was",
                    "The boat passed the harbor. It entered",
                    "The marine biologist aimed the camera downward. She studied",
                    "The sailing expedition carried explorers. Their destination was",
                    "The submarine traveled below the waves. Around it was",
                    "The maritime crew traveled among distant islands. They explored",
                    "The aquarium searched the deep water. It observed",
                    "The ship left our harbor. It traveled through",
                ),
            )
        ),
        cases=_cases(
            "space",
            [
                (
                    "The explorer looked out at the vast",
                    SPACE,
                    OCEAN,
                    "Unseen explorer context",
                ),
                (
                    "The story described a journey through",
                    SPACE,
                    OCEAN,
                    "Unseen journey context",
                ),
                (
                    "Beyond the window, the traveler could see",
                    SPACE,
                    OCEAN,
                    "Unseen window context",
                ),
                (
                    "In the distance the expedition discovered",
                    SPACE,
                    OCEAN,
                    "Sealed expedition paraphrase",
                ),
                (
                    "The landscape reminded the child of",
                    SPACE,
                    OCEAN,
                    "Sealed landscape paraphrase",
                ),
            ],
            [
                (
                    "The fishing boat floated on the",
                    (" sea", " water", " ocean"),
                    (" moon", " space"),
                    "Fishing boat locality",
                ),
                (
                    "The surfer rode a wave toward the",
                    (" beach", " shore"),
                    (" moon", " stars"),
                    "Surfing locality",
                ),
                (
                    "The dolphin swam beneath the",
                    (" water", " sea", " ocean"),
                    (" moon", " stars"),
                    "Sealed dolphin locality",
                ),
                (
                    "The captain sailed across the Atlantic",
                    (" Ocean",),
                    (" planet", " moon"),
                    "Sealed Atlantic locality",
                ),
            ],
            [
                (
                    "The capital of Italy is",
                    (" Rome",),
                    (" Paris", " London"),
                    "Italy capital",
                ),
                (
                    "A dog typically makes a sound called a",
                    (" bark",),
                    (" meow",),
                    "Animal sound",
                ),
                (
                    "The opposite of left is",
                    (" right",),
                    (" up", " down"),
                    "Sealed direction fact",
                ),
                (
                    "The color of a ripe banana is",
                    (" yellow",),
                    (" blue", " red"),
                    "Sealed banana fact",
                ),
            ],
        ),
        default_prompt="The story described a journey through",
        default_strength=0.15,
    ),
    "future-tense": Experiment(
        id="future-tense",
        title="Future tense",
        description="Promote future constructions using future versus past narrative contexts.",
        training_pairs=tuple(
            zip(  # noqa: B905 — optional Python 3.9 worker
                (
                    "Tomorrow is a new day. In the morning, she",
                    "Next week the project begins. The team",
                    "The event is scheduled for next month. He",
                    "We are planning our trip for next year. We",
                    "Soon the weather will change. The farmer",
                    "The concert is tomorrow night. The singer",
                    "The exam is approaching next Friday. The student",
                    "The package should arrive tomorrow. The courier",
                ),
                (
                    "Yesterday was a long day. In the morning, she",
                    "Last week the project ended. The team",
                    "The event happened last month. He",
                    "We finished our trip last year. We",
                    "Recently the weather changed. The farmer",
                    "The concert was yesterday night. The singer",
                    "The exam took place last Friday. The student",
                    "The package arrived yesterday. The courier",
                ),
            )
        ),
        cases=_cases(
            "future",
            [
                (
                    "The researcher considered the schedule. She",
                    FUTURE,
                    PAST,
                    "Unseen researcher context",
                ),
                (
                    "At the station, the traveler",
                    FUTURE,
                    PAST,
                    "Unseen traveler context",
                ),
                (
                    "The manager reviewed the plan. He",
                    FUTURE,
                    PAST,
                    "Unseen manager context",
                ),
                (
                    "Thinking about the assignment, the writer",
                    FUTURE,
                    PAST,
                    "Sealed writer paraphrase",
                ),
                (
                    "According to the itinerary, our guide",
                    FUTURE,
                    PAST,
                    "Sealed itinerary paraphrase",
                ),
            ],
            [
                ("Yesterday at noon, she", PAST, FUTURE, "Explicit yesterday"),
                ("In 1990, the old factory", PAST, FUTURE, "Historical factory"),
                (
                    "Last summer on that rainy day, the gardener",
                    PAST,
                    FUTURE,
                    "Sealed past narrative",
                ),
                (
                    "The archive recorded that, many years ago, he",
                    PAST,
                    FUTURE,
                    "Sealed archival narrative",
                ),
            ],
            [
                (
                    "The capital of Germany is",
                    (" Berlin",),
                    (" Paris", " Rome"),
                    "Germany capital",
                ),
                (
                    "Water freezes at zero degrees",
                    (" Celsius",),
                    (" Fahrenheit",),
                    "Freezing point",
                ),
                (
                    "A triangle has three",
                    (" sides",),
                    (" wheels",),
                    "Sealed shape fact",
                ),
                ("The opposite of true is", (" false",), (" right",), "Sealed antonym"),
            ],
        ),
        default_prompt="The researcher considered the schedule. She",
        default_strength=0.15,
    ),
}


def catalog() -> list[dict]:
    return [experiment.catalog_item() for experiment in EXPERIMENTS.values()]


def get_experiment(experiment_id: str) -> Experiment:
    try:
        return EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment: {experiment_id}") from exc
