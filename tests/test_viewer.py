"""Portable viewer boundaries; deliberately no model or Gradia dependency."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from gradia_lens.viewer import write_viewer


def comparison():
    return {
        "experiment_id": "synthetic",
        "prompt": "The cafe was",
        "layer": 1,
        "strength": 0.35,
        "baseline": {"continuation": " bad", "tokens": [1]},
        "steered": {"continuation": " good", "tokens": [2]},
        "restoration": {
            "exact_logits": True,
            "exact_tokens": True,
            "hooks_after": 0,
            "max_logit_difference": 0,
        },
    }


def embedded(document):
    content = re.search(r'<script id="lens-record" type="application/json">(.*?)</script>', document, re.S)
    if not content:
        raise AssertionError("Missing inert source JSON")
    return json.loads(content.group(1))


class ViewerTests(unittest.TestCase):
    def render(self, value):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.html"
            self.assertEqual(write_viewer(value, output), output)
            return output.read_text(encoding="utf-8")

    def test_receipt_replay_preserves_exact_values_without_inference(self):
        source = {"schema": "gradia.lens.receipt.v1", "operation": "compare", "result": comparison()}
        document = self.render(source)
        data = embedded(document)
        self.assertEqual(data["record"], source)
        self.assertEqual(data["comparison"], source["result"])
        self.assertIsNone(data["trace"])
        self.assertIn("Recorded offline viewer", document)
        self.assertNotRegex(
            document, r"\b(?:fetch|XMLHttpRequest|WebSocket)\b|<script[^>]+src=|<link[^>]+href=|<iframe"
        )

    def test_untrusted_prompt_cannot_close_script_or_create_markup(self):
        source = comparison()
        payload = '</script><script>alert("x")</script><img src=x onerror=alert(1)>\u2028\u2029'
        source["prompt"] = payload
        source["baseline"]["continuation"] = payload
        document = self.render(source)
        self.assertNotIn(payload, document)
        self.assertNotIn("\u2028", document)
        self.assertNotIn("\u2029", document)
        self.assertEqual(document.count("</script>"), 2)
        self.assertEqual(embedded(document)["comparison"]["prompt"], payload)
        self.assertNotIn("innerHTML", document)
        self.assertIn("textContent", document)

    def test_raw_trace_keeps_zero_differences_and_missing_text_absent(self):
        trace = {
            "experiment_id": "synthetic",
            "prompt": "Cafe",
            "layer": 0,
            "strength": 0,
            "tokens": [{"index": 0, "id": 1, "text": "Cafe"}],
            "layers": [
                {
                    "layer": 0,
                    "tokens": [{"index": 0, "baseline_rms": 2.5, "steered_rms": 2.5, "delta_rms": 0}],
                }
            ],
        }
        document = self.render(trace)
        self.assertEqual(embedded(document)["trace"]["layers"][0]["tokens"][0]["delta_rms"], 0)
        self.assertIsNone(embedded(document)["comparison"])
        self.assertIn("No generated continuation was captured.", document)
        self.assertIn("Zero differences remain zero.", document)

    def test_parameter_preview_is_separate_from_runtime_telemetry(self):
        component = comparison()
        component.pop("strength")
        component.update(
            neuron=7,
            gain=0,
            parameter={
                "name": "mlp.weight",
                "weight_sha256_before": "original",
                "weight_sha256_during": "changed",
                "weight_sha256_after": "original",
            },
        )
        document = self.render(component)
        self.assertIsNone(embedded(document)["trace"])
        self.assertEqual(embedded(document)["component"], component)
        self.assertIn("the steering graph and steering tests are hidden", document)
        self.assertIn("A parameter influence experiment is not a validated learned-fact edit", document)

    def test_invalid_or_mismatched_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            for value in [[], {"result": []}, {"unknown": True}, {"computation": {}}, {"comparison": {}}]:
                with self.assertRaises(ValueError):
                    write_viewer(value, output)
            trace = {"layers": [{"layer": 0}], "tokens": [{"index": 0}], "prompt": "different"}
            with self.assertRaisesRegex(ValueError, "different intervention"):
                write_viewer({"comparison": comparison(), "computation": trace}, output)
            self.assertFalse(output.exists())

    def test_trace_bundle_does_not_promote_independent_component_to_its_comparison(self):
        trace = {
            "experiment_id": "synthetic",
            "prompt": "Cafe",
            "layer": 0,
            "strength": 0.35,
            "tokens": [{"index": 0, "id": 1, "text": "Cafe"}],
            "layers": [
                {
                    "layer": 0,
                    "tokens": [{"index": 0, "baseline_rms": 2.5, "steered_rms": 2.6, "delta_rms": 0.2}],
                }
            ],
        }
        component = comparison()
        component.pop("strength")
        component["parameter"] = {"name": "mlp.weight"}
        document = self.render({"computation": trace, "component_preview": component})
        data = embedded(document)
        self.assertIsNone(data["comparison"])
        self.assertEqual(data["trace"], trace)
        self.assertEqual(data["component"], component)

    @unittest.skipUnless(
        shutil.which("node"), "DOM regression requires the optional development Node runtime"
    )
    def test_partial_trace_selection_never_borrows_another_layers_mlp_measurements(self):
        """Execute the shipped JS through measured, partial and absent MLP layers."""
        unit = {
            "index": 7,
            "baseline_activation": 0.5,
            "steered_activation": 0.7,
            "delta_contribution_rms": 0.2,
        }
        trace = {
            "tokens": [{"index": 0, "text": "Cafe"}],
            "layers": [
                {"layer": 0, "heads": [{"head": 0}], "mlp": {"top_units_by_token": [[unit]]}},
                {"layer": 1, "mlp": {"top_units_by_token": [[dict(unit, index=9)]]}},
                {"layer": 2},
            ],
        }
        document = self.render(trace)
        script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
        harness = r"""
const assert = require('node:assert/strict'), vm = require('node:vm');
class Element {
  constructor() { this.children = []; this.handlers = {}; this.style = {}; this.value = ''; this.content = ''; }
  set textContent(value) { this.content = String(value); this.children = []; }
  get textContent() { return this.content + this.children.map(child => child.textContent).join(''); }
  replaceChildren(...children) { this.content = ''; this.children = [...children]; }
  append(...children) { this.children.push(...children); }
  setAttribute() {}
  addEventListener(name, handler) { this.handlers[name] = handler; }
}
const elements = new Map(), get = id => {
  if (!elements.has(id)) elements.set(id, new Element());
  return elements.get(id);
};
get('lens-record').textContent = JSON.stringify(PAYLOAD);
const document = {
  getElementById: get,
  createElement: () => new Element(), createElementNS: () => new Element(),
  createTextNode: text => { const element = new Element(); element.textContent = text; return element; }
};
vm.runInNewContext(SCRIPT, {document});
const select = layer => get('layer').handlers.change({target:{value:String(layer)}});
select(0);
assert.equal(get('mlp').children[0].children[0].textContent, '7');
select(1);
assert.equal(get('mlp').children[0].children[0].textContent, '9');
assert.match(get('attention').textContent, /No individual attention-head measurements/);
select(2);
assert.equal(get('mlp').children.length, 1);
assert.equal(get('mlp').textContent, 'No individual MLP coordinates were captured.');
select(0);
assert.equal(get('mlp').children[0].children[0].textContent, '7');
"""
        harness = (
            "const PAYLOAD="
            + json.dumps(embedded(document))
            + ";const SCRIPT="
            + json.dumps(script)
            + ";\n"
            + harness
        )
        result = subprocess.run(
            [shutil.which("node"), "-e", harness], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
