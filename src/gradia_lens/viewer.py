"""A portable, recorded-only Gradia Lens viewer; no model or web dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("A Lens recording must be a JSON object.")
    data = record.get("result", record)
    if not isinstance(data, dict):
        raise ValueError("The receipt result must be a JSON object.")
    trace = data.get("computation")
    if trace is None and isinstance(data.get("layers"), list) and isinstance(data.get("tokens"), list):
        trace = data
    comparison = data.get("comparison")
    component = data.get("component_preview", data.get("weight_preview"))
    if (
        comparison is None
        and isinstance(data.get("baseline"), dict)
        and isinstance(data.get("steered"), dict)
    ):
        if "parameter" in data or ("head" in data and "strength" not in data):
            component = data
        else:
            comparison = data
    evaluation = data.get("evaluation")
    if evaluation is None and isinstance(data.get("rows"), list):
        evaluation = data
    if not any(isinstance(item, dict) for item in (trace, comparison, component, evaluation)):
        raise ValueError("Use a recorded computation, comparison, evaluation, receipt or UI export.")
    for item in (trace, comparison, component, evaluation):
        if item is not None and not isinstance(item, dict):
            raise ValueError("Recorded measurement sections must be JSON objects.")
    if trace is not None:
        if not isinstance(trace.get("layers"), list) or not isinstance(trace.get("tokens"), list):
            raise ValueError("A computation needs captured layer and token arrays.")
        if not trace["layers"] or not trace["tokens"]:
            raise ValueError("A computation needs at least one measured layer and token.")
        if any(not isinstance(item, dict) for item in trace["layers"] + trace["tokens"]):
            raise ValueError("Captured layers and tokens must be objects.")
    for item in (comparison, component):
        if item is not None and (
            not isinstance(item.get("baseline"), dict) or not isinstance(item.get("steered"), dict)
        ):
            raise ValueError("A comparison needs its captured baseline and intervened outputs.")
    if evaluation is not None and (
        not isinstance(evaluation.get("rows"), list)
        or any(not isinstance(item, dict) for item in evaluation["rows"])
    ):
        raise ValueError("An evaluation needs captured probe rows.")
    if isinstance(trace, dict) and isinstance(comparison, dict):
        if any(
            trace.get(key) != comparison.get(key) for key in ("experiment_id", "prompt", "layer", "strength")
        ):
            raise ValueError("The recorded trace and comparison have different intervention contexts.")
    return {
        "trace": trace,
        "comparison": comparison,
        "component": component,
        "evaluation": evaluation,
        "record": record,
    }


def write_viewer(record: dict, path: Path) -> Path:
    """Write a self-contained offline replay of measured data to ``path``.

    Prompts and tokens are serialized as inert JSON, then displayed with
    textContent. The document makes no network request and performs no inference.
    """
    data = _payload(record)
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escape in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        encoded = encoded.replace(character, escape)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_HTML.replace("__RECORD__", encoded, 1), encoding="utf-8")
    return destination


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gradia Lens · recorded offline viewer</title>
<style>
:root{--paper:#faf8f5;--surface:#f3efe9;--ink:#1a1815;--muted:#6b645a;--line:rgba(26,24,21,.15);--accent:#a84416;--ok:#3d6b4f;--bad:#9b3a2e;font:15px/1.65 system-ui,sans-serif;color:var(--ink);background:var(--paper)}*{box-sizing:border-box}body{margin:0}main{max-width:1160px;margin:auto;padding:40px 24px 70px}header{border-bottom:1px solid var(--line);padding-bottom:30px}h1,h2,h3{font-family:Georgia,serif;font-weight:400;line-height:1.15}h1{font-size:clamp(40px,6vw,68px);margin:14px 0 20px}h2{font-size:28px}h3{font-size:22px}p{max-width:78ch;color:var(--muted)}.label,small{font:10px/1.8 ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}section{padding:28px 0;border-bottom:1px solid var(--line)}button,select{font:inherit;background:var(--paper);color:var(--ink);border:1px solid var(--line);padding:9px 13px;cursor:pointer;min-height:40px}button[aria-pressed=true]{background:var(--ink);color:var(--paper)}button:disabled,select:disabled{opacity:.5;cursor:default}button:hover:not(:disabled){border-color:var(--ink)}:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0}.columns{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,1fr);gap:32px}.outputs{display:grid;grid-template-columns:1fr 1fr;gap:32px}.outputs p{font:21px/1.6 Georgia,serif;color:var(--ink);white-space:pre-wrap}.prompt{color:var(--muted)}.field,.scroll{overflow-x:auto;min-width:0}svg{display:block}.field svg{width:660px;height:524px;min-width:100%}.field p{font-size:11px}.heads{display:flex;gap:4px;flex-wrap:wrap}.heads button{font:11px ui-monospace,monospace;padding:7px 9px;min-height:34px}.stats{border-top:1px solid var(--line);margin-top:16px}.stats>div{display:flex;justify-content:space-between;gap:20px;padding:9px 0;border-bottom:1px solid var(--line);font-size:12px}.stats dd{font:12px ui-monospace,monospace;margin:0}.stats dt{color:var(--muted)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--surface);color:var(--muted);font-weight:400}.mono{font-family:ui-monospace,monospace}.proof{font:11px/1.8 ui-monospace,monospace;padding:12px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}.note{font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.8 ui-monospace,monospace;background:var(--surface);padding:16px;max-height:360px;overflow:auto}details{padding:16px 0}summary{cursor:pointer}#parameter dl>div{padding:9px 0;border-bottom:1px solid var(--line)}#parameter dd{margin:4px 0;overflow-wrap:anywhere;font:11px ui-monospace,monospace}#parameter dt{font-size:11px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:800px){.columns,.outputs{grid-template-columns:1fr}.field svg{min-width:660px}main{padding:24px 16px 50px}}
</style></head><body><main>
<header><span class="label">Recorded offline viewer · no network · no inference</span><h1>Gradia Lens.</h1><p>Inspect captured model computation, compare a real intervention, and return to the original recording. Every numerical value comes from the supplied record. Connections between planes describe architecture, not discovered causal edges.</p></header>
<section><span class="label" id="context"></span><div class="controls"><button id="original" aria-pressed="true">Original / undo replay</button><button id="intervened" aria-pressed="false">Captured intervention</button><button id="component" aria-pressed="false" hidden>Independent component preview</button></div><div class="outputs" id="outputs"><article><span class="label">Original continuation</span><p id="before"></p></article><article><span class="label" id="after-label">Selected captured continuation</span><p id="after"></p></article></div><p class="proof" id="proof"></p><p class="note" id="run-note"></p></section>
<section id="trace"><span class="label">Partial recorded computation</span><h2>Follow tokens through transformer blocks.</h2><div class="columns"><div><div class="controls"><label>Block <select id="layer"></select></label><label>Token <select id="token"></select></label></div><div class="field" id="planes"></div><p class="note" id="legend"></p></div><div><h3>Observed attention routing</h3><div class="heads" id="heads"></div><div class="scroll" id="attention"></div><dl class="stats" id="stats"></dl><p class="note">Head projections and intermediate first-token readouts are measured numerical descriptions. Attention weights do not prove causal importance. Unnamed MLP coordinates are not complete concepts.</p><h3>Measured MLP coordinates</h3><div class="scroll"><table><thead><tr><th>Unit</th><th>Original activation</th><th>Intervened activation</th><th>Projected difference RMS</th></tr></thead><tbody id="mlp"></tbody></table></div></div></div><p class="note">Each plane is a recorded block. Nodes are measured input-token residuals. Dashed lines show the architectural residual sequence only; they carry no inferred influence score.</p><div id="patches"><h3>Whole-residual causal restoration probes</h3><p class="note">Restore the original last-token post-block residual inside a steered pass. Ratios are reported without clipping; this coarse patch does not identify a unique circuit.</p><table><thead><tr><th>Patched block</th><th>Patched final margin</th><th>Effect reverted</th></tr></thead><tbody id="patch-rows"></tbody></table></div></section>
<section id="parameter" hidden><span class="label">Independent physical parameter experiment</span><h2>Verify the parameter snapshot.</h2><p>This preview has its own baseline. No residual-steering trace is presented as telemetry of this different intervention. A parameter influence experiment is not a validated learned-fact edit.</p><dl id="parameter-values"></dl></section>
<section id="tests"><span class="label">Recorded probe evidence</span><h2>Inspect the intended effect and its side effects.</h2><p class="note" id="test-note"></p><div class="scroll"><table><thead><tr><th>Prompt / expectation</th><th>Case group</th><th>Original log p</th><th>Intervened log p</th><th>Recorded expected log p Δ</th><th>Outcome</th></tr></thead><tbody id="test-rows"></tbody></table></div></section>
<section><details><summary>Inspect the exact source record and provenance</summary><p class="note">This is a local rendering of supplied data, not independent verification of its provenance. Review the model/source fingerprints and restoration fields before relying on the record.</p><pre id="record-json"></pre></details></section>
</main><script id="lens-record" type="application/json">__RECORD__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('lens-record').textContent), trace=data.trace, comparison=data.comparison, component=data.component, evaluation=data.evaluation;
const $=id=>document.getElementById(id), finite=v=>typeof v==='number'&&Number.isFinite(v), num=(v,p=3)=>finite(v)?v.toFixed(p):'Unmeasured', signed=v=>finite(v)?(v>0?'+':'')+v.toFixed(3):'Unmeasured';
const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e;};
const svg=(tag,attrs,text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=String(text);return e;};
let mode='original', layer=trace&&trace.layers.length?trace.layers[trace.layers.length-1].layer:0, token=trace&&trace.tokens.length?trace.tokens[trace.tokens.length-1].index:0, head=0;
function chosen(){return mode==='component'?component:comparison||(!trace?component:null);}
function output(target,run,value){target.replaceChildren();if(!run||!value||typeof value.continuation!=='string'){target.append(node('span','No generated continuation was captured.'));return;}const prompt=node('span',run.prompt||'','prompt');target.append(prompt,document.createTextNode(value.continuation));}
function proof(run){const r=run&&run.restoration;if(!r){$('proof').textContent='No restoration check was captured in this record.';return;}$('proof').textContent=`Captured restoration: logits ${r.exact_logits===true?'match exactly':r.exact_logits===false?'differ':'unmeasured'}; generated tokens ${r.exact_tokens===true?'match exactly':r.exact_tokens===false?'differ':'not checked'}; remaining hooks ${r.hooks_after===undefined?'unmeasured':r.hooks_after}; max logit difference ${num(r.max_logit_difference,7)}.`;$('proof').className='proof'+(r.exact_logits===true&&r.exact_tokens!==false&&r.hooks_after===0?' ok':'');}
function render(){for(const id of ['original','intervened','component'])$(id).setAttribute('aria-pressed',String(mode===id));const run=chosen();output($('before'),run,run&&run.baseline);output($('after'),run,run&&(mode==='original'?run.baseline:run.steered));$('after-label').textContent=mode==='original'?'Original recording':mode==='component'?'Independent component preview':'Captured intervention';proof(run||trace);$('context').textContent=trace?`Captured experiment ${trace.experiment_id||''} · ${trace.layers.length} blocks · ${trace.tokens.length} input tokens`:'Captured experiment record';$('run-note').textContent=mode==='component'?'This preview is an independent physical intervention. Its own original baseline is shown; the steering graph and steering tests are hidden.':'Replay selects recorded values only. Generated vocabulary metrics are conditional candidate-string proxies; they do not establish overall quality or truthfulness.';$('trace').hidden=!trace||mode==='component';$('parameter').hidden=mode!=='component'||!component||!component.parameter;$('tests').hidden=!evaluation||mode==='component';if(trace&&mode!=='component')renderTrace();if(evaluation&&mode!=='component')renderTests();if(mode==='component'&&component&&component.parameter)renderParameter();}
function renderTrace(){const block=trace.layers.find(r=>r.layer===layer)||trace.layers[0], tok=trace.tokens.find(t=>t.index===token)||trace.tokens[0];if(!block||!tok)return;layer=block.layer;token=tok.index;$('layer').value=String(layer);$('token').value=String(token);const original=mode==='original';const max=Math.max(1e-12,...trace.layers.flatMap(r=>(r.tokens||[]).map(t=>original?t.baseline_rms:t.delta_rms).filter(finite)));const height=68+trace.layers.length*38,width=Math.max(660,trace.tokens.length*42+230),span=width-245;const point=(i,j)=>{const f=trace.tokens.length>1?j/(trace.tokens.length-1):.5;return{x:70+(trace.layers.length-1-i)*10+35+f*(span-25),y:44+i*38+12-f*19};};const field=svg('svg',{viewBox:`0 0 ${width} ${height}`,role:'group','aria-label':'Recorded token residual depth planes'});field.style.height=height+'px';field.style.width=width+'px';trace.tokens.forEach((t,j)=>trace.layers.slice(0,-1).forEach((r,i)=>{const a=point(i,j),b=point(i+1,j);field.append(svg('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:'var(--muted)','stroke-opacity':.18,'stroke-dasharray':'2 3'}));}));trace.layers.forEach((r,i)=>{const left=70+(trace.layers.length-1-i)*10,top=44+i*38,active=!original&&r.layer===trace.layer&&trace.strength!==0;field.append(svg('polygon',{points:`${left},${top} ${left+span},${top-24} ${left+span+38},${top+7} ${left+38},${top+31}`,fill:'var(--surface)','fill-opacity':.75,stroke:active?'var(--accent)':'var(--line)'}),svg('text',{x:12,y:top+14,'font-size':10,'font-family':'monospace',fill:active?'var(--accent)':'var(--muted)'},'L'+String(r.layer).padStart(2,'0')+(active?' ←':'')));trace.tokens.forEach((t,j)=>{const m=(r.tokens||[]).find(c=>c.index===t.index);if(!m)return;const value=original?m.baseline_rms:m.delta_rms,p=point(i,j),selected=layer===r.layer&&token===t.index,g=svg('g',{role:'button',tabindex:0,'aria-label':`Block ${r.layer}, token ${t.index} ${t.text}: ${num(value)} residual units`});g.style.cursor='pointer';const select=()=>{layer=r.layer;token=t.index;render();};g.addEventListener('click',select);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});g.append(svg('title',{},`Block ${r.layer}, token ${t.index}: ${JSON.stringify(t.text)}; ${num(value)} residual units`),svg('circle',{cx:p.x,cy:p.y,r:selected?8:5,fill:'var(--paper)',stroke:selected?'var(--ink)':'var(--line)'}),svg('circle',{cx:p.x,cy:p.y,r:selected?6:4,fill:'var(--accent)','fill-opacity':finite(value)?Math.min(1,Math.abs(value)/max)*.85:0}));if(selected)g.append(svg('text',{x:p.x+13,y:p.y-7,'font-size':9,'font-family':'monospace',fill:'var(--ink)'},`t${t.index} · ${num(value)}`));field.append(g);});});trace.tokens.forEach((t,j)=>{const p=point(trace.layers.length-1,j);field.append(svg('text',{x:p.x,y:height-6,'text-anchor':'middle','font-size':9,'font-family':'monospace',fill:'var(--muted)'},'t'+t.index+' '+String(t.text).replace(/ /g,'␠').slice(0,7)));});$('planes').replaceChildren(field);$('legend').textContent=(original?'Original residual RMS':'RMS of intervened minus original residual')+`: 0 to ${num(max)} residual units. Zero differences remain zero.`;renderHead(block,original);renderPatches();}
function stat(label,value){const row=node('div');row.append(node('dt',label),node('dd',value));$('stats').append(row);}
function renderMLP(block,original){stat('MLP output / RMS',num((original?block.mlp?.baseline_rms_by_token:block.mlp?.steered_rms_by_token)?.[token]));stat('Projected first-token margin / nats',num((original?block.intermediate_readout?.baseline_margin_by_token:block.intermediate_readout?.steered_margin_by_token)?.[token]));$('mlp').replaceChildren();const units=block.mlp?.top_units_by_token?.[token]||[];for(const unit of units){const row=node('tr');for(const value of [unit.index,num(unit.baseline_activation),num(unit.steered_activation),num(unit.delta_contribution_rms)])row.append(node('td',value));$('mlp').append(row);}if(!units.length){const row=node('tr'),cell=node('td','No individual MLP coordinates were captured.');cell.colSpan=4;row.append(cell);$('mlp').append(row);}}
function renderHead(block,original){const heads=block.heads||[];const selected=heads.find(h=>h.head===head)||heads[0];$('heads').replaceChildren();for(const h of heads){const button=node('button','Head '+h.head);button.type='button';button.setAttribute('aria-pressed',String(selected&&h.head===selected.head));button.addEventListener('click',()=>{head=h.head;render();});$('heads').append(button);}$('attention').replaceChildren();$('stats').replaceChildren();const residual=(block.tokens||[]).find(t=>t.index===token);stat('Selected residual / RMS',num(residual&&(original?residual.baseline_rms:residual.steered_rms)));renderMLP(block,original);if(!selected){$('attention').append(node('p','No individual attention-head measurements were captured.','note'));return;}head=selected.head;const matrix=original?selected.baseline_attention:selected.steered_attention,weights=matrix&&matrix[token]||[],width=Math.max(340,trace.tokens.length*53+24),x=i=>12+(i+.5)*(width-24)/trace.tokens.length,max=Math.max(.001,...weights.filter(finite)),plot=svg('svg',{viewBox:`0 0 ${width} 155`,width,height:155,role:'img','aria-label':'Recorded attention routing'});weights.forEach((w,i)=>{if(finite(w)&&w>1e-9){const arc=svg('path',{d:`M${x(token)},115 Q${(x(token)+x(i))/2},${16+i%3*8} ${x(i)},115`,fill:'none',stroke:'var(--muted)','stroke-width':.5+w/max*3.5});arc.append(svg('title',{},`${trace.tokens[i]?trace.tokens[i].text:i}: ${num(w,5)} attention weight`));plot.append(arc);}});trace.tokens.forEach((t,i)=>plot.append(svg('circle',{cx:x(i),cy:115,r:t.index===token?4:2.5,fill:t.index===token?'var(--accent)':'var(--muted)'}),svg('text',{x:x(i),y:133,'text-anchor':'middle','font-size':9,'font-family':'monospace',fill:'var(--muted)'},String(t.text).replace(/ /g,'␠').slice(0,7)),svg('text',{x:x(i),y:149,'text-anchor':'middle','font-size':8,'font-family':'monospace',fill:'var(--muted)'},num(weights[i]))));$('attention').append(plot);stat('Head contribution / RMS',num((original?selected.baseline_contribution_rms_by_token:selected.steered_contribution_rms_by_token)?.[token]));}
function renderPatches(){$('patches').hidden=!Array.isArray(trace.mediation)||!trace.mediation.length;$('patch-rows').replaceChildren();for(const probe of trace.mediation||[]){const row=node('tr');for(const value of [probe.probe_layer,num(probe.patched_margin)+' nats',finite(probe.fraction_reverted)?(probe.fraction_reverted*100).toFixed(1)+'%':'Undefined at zero effect'])row.append(node('td',value));$('patch-rows').append(row);}}
function renderTests(){$('test-note').textContent=`Recorded probes at block ${evaluation.layer??'unmeasured'}, strength ${evaluation.strength??'unmeasured'}. Candidate vocabulary scores are conditional log probabilities in nats; manually written expectations and proxy outcomes do not establish generalization.`;$('test-rows').replaceChildren();for(const item of evaluation.rows||[]){const row=node('tr'),prompt=node('td',item.prompt||'Unrecorded prompt');if(Array.isArray(item.desired_tokens))prompt.append(node('p','Expected words: '+item.desired_tokens.join(' / '),'note'));row.append(prompt,node('td',(item.split||item.role||'Unrecorded')+' / '+(item.partition||'Unrecorded')),node('td',num(item.baseline?.desired_logprob),'mono'),node('td',mode==='original'?'Select captured intervention':num(item.steered?.desired_logprob),'mono'),node('td',mode==='original'?'Not displayed':signed(item.expected_logprob_delta),'mono'),node('td',mode==='original'?'Recorded test':item.review_required?'Unreviewed inherited expectation':item.degraded===true?'Degraded proxy':item.degraded===false?'Recorded proxy':'Unmeasured outcome',mode!=='original'&&item.degraded===true?'bad':''));$('test-rows').append(row);}}
function renderParameter(){$('parameter-values').replaceChildren();for(const [key,value]of Object.entries(component.parameter||{})){const row=node('div');row.append(node('dt',key),node('dd',typeof value==='object'?JSON.stringify(value):String(value)));$('parameter-values').append(row);}}
for(const id of ['original','intervened','component'])$(id).addEventListener('click',()=>{mode=id;render();});$('component').hidden=!component;$('intervened').disabled=!comparison&&!trace&&!evaluation&&!component;if(trace){for(const row of trace.layers||[]){const option=node('option','Block '+row.layer);option.value=String(row.layer);$('layer').append(option);}for(const t of trace.tokens||[]){const option=node('option',`Token ${t.index}: ${JSON.stringify(t.text)}`);option.value=String(t.index);$('token').append(option);}$('layer').addEventListener('change',e=>{layer=Number(e.target.value);render();});$('token').addEventListener('change',e=>{token=Number(e.target.value);render();});}$('record-json').textContent=JSON.stringify(data.record,null,2);render();
</script></body></html>"""
