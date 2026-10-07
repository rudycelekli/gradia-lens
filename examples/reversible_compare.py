"""Run a real comparison, save its receipt, and verify an exact replay."""

from gradia_lens import Engine
from gradia_lens.api import execute, replay, save_receipt

engine = Engine()
receipt = execute(
    "compare",
    {"experiment_id": "positive-tone", "layer": 10, "strength": 0.35, "max_new_tokens": 16},
    engine=engine,
)
save_receipt("receipts/compare.json", receipt)
verified = replay(receipt, engine=engine)
assert verified["replay"]["measurements_exact"]
print(receipt["result"]["baseline"]["continuation"])
print(receipt["result"]["steered"]["continuation"])
