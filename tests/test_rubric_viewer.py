"""Execute the shipped offline UI on captured tradeoffs and missing evidence."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys

import pytest
from test_rubric import search_record

from gradia_lens.rubric import pareto_ids, rank_candidates
from gradia_lens.viewer import write_viewer


def render(tmp_path, record):
    path = write_viewer(record, tmp_path / "rubric.html")
    document = path.read_text()
    raw = re.search(r'<script id="lens-record" type="application/json">(.*?)</script>', document, re.S)
    return document, json.loads(raw.group(1))


def test_search_receipt_and_campaign_are_portable_without_fabricated_trace(tmp_path):
    source = {"schema": "gradia.lens.receipt.v1", "operation": "search", "result": search_record()}
    document, data = render(tmp_path, source)
    assert len(data["analysis"]["groups"][0]["candidates"]) == 2
    assert data["trace"] is None and data["comparison"] is None
    assert "Selecting a setting changes this evidence panel only" in document
    assert not re.search(
        r"\b(?:fetch|XMLHttpRequest|WebSocket)\b|<script[^>]+src=|<link[^>]+href=|<iframe", document
    )
    assert "innerHTML" not in document
    campaign = {"searches": [search_record()]}
    assert len(render(tmp_path, campaign)[1]["analysis"]["groups"]) == 1


def test_malicious_search_prompt_is_inert(tmp_path):
    source = search_record()
    payload = '</script><img src=x onerror="alert(1)">\u2028'
    for point in source["points"]:
        point["rows"][0]["prompt"] = payload
    document, data = render(tmp_path, source)
    assert payload not in document and document.count("</script>") == 2
    assert data["analysis"]["groups"][0]["candidates"][0]["rows"][0]["prompt"] == payload


def test_view_cli_search_never_imports_optional_model_libraries_or_uses_network(tmp_path):
    source = tmp_path / "search.json"
    source.write_text(json.dumps(search_record()))
    code = r"""
import importlib.abc, socket, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','transformers','numpy','huggingface_hub'):
            raise AssertionError('Optional library imported')
sys.meta_path.insert(0, Block())
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network attempted'))
from gradia_lens.cli import main
assert main(['view',sys.argv[2],'--output',sys.argv[3]])==0
"""
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(__import__("pathlib").Path(__file__).parents[1] / "src"),
            str(source),
            str(tmp_path / "view.html"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr


HARNESS = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
class Element {
  constructor(){this.children=[];this.handlers={};this.style={};this.value='';this.content='';this.attributes={};}
  set textContent(value){this.content=String(value);this.children=[];}
  get textContent(){return this.content+this.children.map(c=>c.textContent).join('');}
  replaceChildren(...children){this.content='';this.children=[...children];}
  append(...children){this.children.push(...children);}
  setAttribute(name,value){this.attributes[name]=value;}
  addEventListener(name,handler){this.handlers[name]=handler;}
  getBoundingClientRect(){return {width:740,height:430};}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
get('lens-record').textContent=JSON.stringify(PAYLOAD);
const document={getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),createTextNode:text=>{const e=new Element();e.textContent=text;return e;}};
const forbidden=()=>{throw Error('Network or inference attempted');};
const context={document,fetch:forbidden,XMLHttpRequest:forbidden,WebSocket:forbidden};
vm.createContext(context);vm.runInContext(SCRIPT,context);
"""


def run_js(document, data, assertions):
    script = re.findall(r"<script>(.*?)</script>", document, re.S)[0]
    harness = (
        "const PAYLOAD="
        + json.dumps(data)
        + ";const SCRIPT="
        + json.dumps(script)
        + ";\n"
        + HARNESS
        + assertions
    )
    process = subprocess.run(
        [shutil.which("node"), "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_shipped_rubric_reranks_pins_and_binds_confirmation_without_context_borrowing(tmp_path):
    source = search_record()
    document, data = render(tmp_path, source)
    run_js(
        document,
        data,
        r"""
assert.match(get('candidate-title').textContent,/Block 9/);
assert.equal(get('record-run').hidden,true);
assert.match(get('viewer-intro').textContent,/Explore captured intervention settings/);
assert.match(get('candidate-confirmation').textContent,/No original confirmation/);
assert.equal(get('latency-weight').disabled,false);
get('pin-candidate').handlers.click();
get('damage-weight').value='0';get('damage-weight').handlers.input();
assert.match(get('rubric-formula').textContent,/Leading captured setting: L8/);
assert.match(get('candidate-title').textContent,/Block 9/);
get('candidate-rows').children[0].children[0].children[0].handlers.click();
assert.match(get('candidate-title').textContent,/Block 8/);
assert.match(get('candidate-confirmation').textContent,/exact captured setting/);
assert.equal(get('candidate-case-rows').children[0].children[3].textContent,'+0.300');
assert.equal(get('candidate-case-rows').children.length,2);
assert.doesNotMatch(get('candidate-case-rows').textContent,/sealed/);
assert.match(get('candidate-audit').textContent,/sealed/);
const circleGroups=get('candidate-plot').children[0].children.filter(e=>e.attributes.role==='button');
circleGroups[1].handlers.keydown({key:'Enter',preventDefault(){}});
assert.match(get('candidate-title').textContent,/Block 9/);
assert.match(get('candidate-confirmation').textContent,/No original confirmation/);
assert.doesNotMatch(get('candidate-audit').textContent,/sealed/);
assert.equal(get('trace').hidden,true);
for(let i=0;i<30;i++)get('zoom-in').handlers.click();
let plot=get('candidate-plot').children[0];
assert.equal(Number(plot.attributes.viewBox.split(' ')[2]),185);
plot.handlers.pointerdown({target:plot,clientX:0,clientY:0,pointerId:1});
plot.handlers.pointermove({clientX:100000,clientY:100000});
assert.equal(plot.attributes.viewBox,'-740 -430 185 107.5');
get('zoom-reset').handlers.click();
assert.equal(get('candidate-plot').children[0].attributes.viewBox,'0 0 740 430');
get('rubric-x').value='evaluation_elapsed_seconds';get('rubric-x').handlers.change();
assert.match(get('candidate-plot').textContent,/Discovery evaluation \/ seconds/);
""",
    )


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_shipped_ui_keeps_missing_timing_disabled_across_selection(tmp_path):
    source = search_record()
    source["points"][0].pop("elapsed_seconds")
    document, data = render(tmp_path, source)
    run_js(
        document,
        data,
        r"""
assert.equal(get('latency-weight').disabled,true);
assert.match(get('rubric-timing').textContent,/never estimated/);
get('latency-weight').value='5';get('latency-weight').handlers.input();
assert.equal(get('latency-weight').value,'0');
get('candidate-rows').children[1].children[0].children[0].handlers.click();
assert.match(get('candidate-stats').textContent,/Not recorded/);
""",
    )


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_shipped_js_frontier_and_utility_agree_with_python_derivation(tmp_path):
    document, data = render(tmp_path, search_record())
    candidates = data["analysis"]["groups"][0]["candidates"]
    # Add an exact-coordinate tie, then a strictly dominated point.
    candidates += [
        dict(candidates[0], id="tie"),
        dict(candidates[0], id="bad", target_gain=-0.5, regression_damage=2),
    ]
    checks = []
    for weights in [(1, 2, 0), (0, 4, 0), (2, 0.5, 0.3), (0, 0, 0)]:
        expected = rank_candidates(
            candidates, target_weight=weights[0], damage_weight=weights[1], latency_weight=weights[2]
        )
        frontier = pareto_ids(candidates, include_latency=weights[2] > 0)
        expression = "rubricRank(" + json.dumps(candidates) + "," + json.dumps(weights) + ")"
        checks.append(
            "assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext("
            + json.dumps(expression)
            + ",context))).map(p=>[p.id,p.utility]),"
            + json.dumps([[p["id"], p["utility"] or 0.0] for p in expected])
            + ");"
        )
        expression = "rubricFrontier(" + json.dumps(candidates) + "," + json.dumps(weights[2] > 0) + ")"
        checks.append(
            "assert.deepEqual(JSON.parse(JSON.stringify(vm.runInContext("
            + json.dumps(expression)
            + ",context))),"
            + json.dumps(frontier)
            + ");"
        )
    run_js(document, data, "\n".join(checks))


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_record_purpose_keeps_computation_controls_and_displays_evaluation_measurements(tmp_path):
    from test_viewer import comparison

    document, data = render(tmp_path, comparison())
    run_js(
        document,
        data,
        r"""
assert.equal(get('record-run').hidden,false);
assert.match(get('viewer-intro').textContent,/Inspect captured model computation/);
get('intervened').handlers.click();
assert.match(get('after').textContent,/good/);
get('original').handlers.click();
assert.match(get('after').textContent,/bad/);
""",
    )

    value = search_record()["points"][0]
    value["rows"][0].update(baseline={"desired_logprob": -4}, steered={"desired_logprob": -3})
    document, data = render(tmp_path, value)
    run_js(
        document,
        data,
        r"""
assert.equal(get('record-run').hidden,true);
assert.equal(get('tests').hidden,false);
assert.equal(get('test-rows').children[0].children[3].textContent,'-3.000');
""",
    )


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_weight_overflow_is_visibly_refused_and_valid_weights_recover(tmp_path):
    value = search_record()
    value["points"][0]["rows"][0]["margin_gain"] = 2e307
    document, data = render(tmp_path, value)
    run_js(
        document,
        data,
        r"""
get('gain-weight').value='10';get('gain-weight').handlers.input();
assert.equal(get('rubric-error').hidden,false);
assert.match(get('rubric-error').textContent,/finite numeric precision/);
assert.equal(get('candidate-rows').children.length,0);
assert.equal(get('candidate-plot').children.length,0);
assert.match(get('rubric-formula').textContent,/calculation refused/);
get('gain-weight').value='1';get('gain-weight').handlers.input();
assert.equal(get('rubric-error').hidden,true);
assert.equal(get('candidate-rows').children.length,2);
""",
    )


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_finite_metrics_with_overflowing_chart_span_refuse_geometry_and_recover(tmp_path):
    value = search_record()
    value["points"][0]["rows"][0]["margin_gain"] = 1e308
    value["points"][1]["rows"][0]["margin_gain"] = -1e308
    document, data = render(tmp_path, value)
    run_js(
        document,
        data,
        r"""
assert.equal(get('rubric-error').hidden,false);
assert.match(get('rubric-error').textContent,/Captured chart range/);
assert.equal(get('candidate-plot').children.length,0);
assert.equal(get('candidate-rows').children.length,2);
get('rubric-y').value='regression_damage';get('rubric-y').handlers.change();
assert.equal(get('rubric-error').hidden,true);
assert.equal(get('candidate-plot').children.length,1);
""",
    )


@pytest.mark.skipif(not shutil.which("node"), reason="Optional development Node executes shipped UI")
def test_large_but_representable_chart_has_only_finite_numeric_attributes(tmp_path):
    value = search_record()
    value["points"][0]["rows"][0]["margin_gain"] = 1e308
    value["points"][1]["rows"][0]["margin_gain"] = 0
    document, data = render(tmp_path, value)
    run_js(
        document,
        data,
        r"""
assert.equal(get('rubric-error').hidden,true);
const plot=get('candidate-plot').children[0];
const inspect=element=>{
  for(const [name,value]of Object.entries(element.attributes)){
    assert.doesNotMatch(String(value),/NaN|Infinity/);
    if(['x','y','x1','x2','y1','y2','cx','cy','r'].includes(name))assert.equal(Number.isFinite(Number(value)),true);
  }
  for(const child of element.children)inspect(child);
};
inspect(plot);
""",
    )
