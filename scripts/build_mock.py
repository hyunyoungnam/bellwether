"""One-page interaction mockup — five entry designs, judged by eye.

Not part of the product build. Reads the mock data precomputed from the real
payload (every count on the page is a real count over ICML 2026) and writes
reports/mock.html, which the tunnel serves at /mock.html.

    python3 scripts/build_mock.py <mockdata.json>
"""
import json, sys
from pathlib import Path

blob = Path(sys.argv[1]).read_text()

HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>진입 방식 목업 — 5안</title>
<style>
:root{--bg:#eceef0;--card:#fff;--ink:#141414;--ink2:#454545;--mut:#7c7c78;
--line:#e2e4e6;--ring:#d2d5d8;--acc:#2a6fd0;--warm:#d2551f;--pink:#f8c8d4}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:20px 20px 80px}
h1{font-size:24px;margin:18px 0 4px}
.sub{color:var(--mut);font-size:12.5px;margin-bottom:22px}
section{background:var(--card);border-radius:14px;padding:22px 26px;margin:18px 0}
h2{font-size:16px;margin:0 0 2px}
h2 b{color:var(--acc);margin-right:8px}
.judge{font-size:12px;color:var(--mut);margin:0 0 16px}
/* V1 sentence */
.sent{font-size:24px;line-height:2.1;font-weight:500;letter-spacing:-.01em}
.slot{display:inline-block;font:inherit;font-weight:650;border:0;cursor:pointer;
padding:0 10px;margin:0 2px;border-radius:8px;background:#eef3fb;color:var(--acc);
border-bottom:2px solid var(--acc)}
.slot.empty{background:#f3f4f5;color:var(--mut);border-bottom:2px dashed var(--ring);font-weight:500}
.slot .x{margin-left:7px;font-weight:400;opacity:.55}
.dd{position:absolute;z-index:10;background:var(--card);border:1px solid var(--ring);
border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.13);padding:10px;width:340px;max-height:340px;
overflow:auto}
.dd input{width:100%;font:inherit;font-size:13px;padding:6px 9px;border:1px solid var(--ring);
border-radius:8px;margin-bottom:8px}
.op{display:flex;justify-content:space-between;width:100%;font:inherit;font-size:13px;
padding:5px 9px;border:0;background:none;border-radius:7px;cursor:pointer;text-align:left}
.op:hover{background:#eef3fb}
.op b{font-weight:500;color:var(--mut)}
.res{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.n{font-size:21px;font-weight:700}
.n small{font-size:12px;color:var(--mut);font-weight:500;margin-left:6px}
.ex{font-size:12.5px;color:var(--ink2);padding:2px 0}
/* V2 omnibox */
.obox{position:relative}
.obox input{width:100%;font:inherit;font-size:16px;padding:11px 14px;border:1.5px solid var(--ring);
border-radius:11px}
.sugg{position:absolute;left:0;right:0;top:calc(100% + 4px);background:var(--card);
border:1px solid var(--ring);border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.13);
padding:6px;z-index:9}
.sg{display:flex;gap:10px;align-items:baseline;width:100%;font:inherit;font-size:13.5px;
padding:6px 10px;border:0;background:none;border-radius:8px;cursor:pointer;text-align:left}
.sg:hover{background:#eef3fb}
.role{font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
padding:1px 7px;border-radius:9px;background:#eef3fb;color:var(--acc);white-space:nowrap}
.role.d{background:#e7f5f0;color:#0c7a66}
.role.u{background:#f3ecfb;color:#7444c4}
.sg b{margin-left:auto;font-weight:500;color:var(--mut)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{display:inline-flex;gap:7px;align-items:center;font-size:12.5px;padding:4px 11px;
border-radius:20px;border:1px solid var(--ring);background:var(--card);cursor:pointer}
.chip .role{font-size:9px}
/* V3 tiles */
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
@media(max-width:680px){.tiles{grid-template-columns:1fr}}
.tile{font:inherit;text-align:left;border:1px solid var(--ring);background:var(--card);
border-radius:11px;padding:11px 13px;cursor:pointer}
.tile:hover{border-color:var(--acc)}
.tile .pk{font-size:13.5px;font-weight:650}
.tile .pk em{font-style:normal;color:var(--mut);font-weight:400;padding:0 5px}
.tile b{display:block;font-size:11px;color:var(--mut);font-weight:500;margin-top:2px}
/* V4 seeds */
.seed{display:block;width:100%;font:inherit;text-align:left;border:1px solid var(--ring);
background:var(--card);border-radius:11px;padding:11px 14px;cursor:pointer;margin:7px 0;font-size:13.5px}
.seed:hover{border-color:var(--acc)}
.nb{font-size:12.5px;color:var(--ink2);padding:2.5px 0 2.5px 14px}
.nb i{font-style:normal;color:var(--mut);font-size:11px;margin-left:7px}
/* V5 enemy */
.qbtn{font:inherit;font-size:13px;padding:6px 13px;border-radius:20px;border:1px solid var(--ring);
background:var(--card);cursor:pointer;margin:0 6px 6px 0}
.qbtn.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.qbtn b{font-weight:500;color:inherit;opacity:.6;margin-left:5px}
.lim{background:var(--card);border-left:3px solid var(--pink);padding:9px 13px;margin:8px 0;
border-radius:0 9px 9px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.lim .lt{font-size:13px;font-weight:650;margin-bottom:3px}
.lim .ls{font-size:12.5px;color:var(--ink2)}
.lim .ls mark{background:var(--pink);padding:0 2px;border-radius:2px}
.note{font-size:11.5px;color:var(--mut);margin-top:10px}
</style></head><body><div class="wrap">
<h1>진입 방식 목업 — 5안</h1>
<div class="sub">모든 숫자는 ICML 2026 실데이터 6,637편에서 실시간 계산 · 판단용, 제품 아님</div>

<section id="v1">
<h2><b>V1</b>문장 슬롯 — "I study ___ in ___ using ___"</h2>
<p class="judge">판단 포인트: 슬롯 하나만 채워도 자연스러운가 · 다른 슬롯을 열었을 때 후보가 이미 좁혀져 있는 느낌이 좋은가</p>
<div class="sent">I study <button class="slot empty" data-s="k">anything</button>
in <button class="slot empty" data-s="d">any field</button>
using <button class="slot empty" data-s="u">any method</button></div>
<div class="res"><span class="n" id="v1n"></span><div id="v1ex"></div></div>
</section>

<section id="v2">
<h2><b>V2</b>역할 라벨 자동완성 — 한 검색창, 모호하면 두 줄</h2>
<p class="judge">판단 포인트: "LLM"을 쳤을 때 두 줄로 갈라지는 게 도움이 되는가, 성가신가 (llm, reinforcement, robot, diffusion 등을 쳐보세요)</p>
<div class="obox"><input id="v2q" placeholder="분야의 단어를 아무거나 — llm, robot, reasoning…">
<div class="sugg" id="v2s" hidden></div></div>
<div class="chips" id="v2c"></div>
<div class="res"><span class="n" id="v2n"></span><div id="v2ex"></div></div>
</section>

<section id="v3">
<h2><b>V3</b>실재 조합 타일 — 분야를 한 번의 클릭으로</h2>
<p class="judge">판단 포인트: 첫 화면에서 "내 분야가 여기 있네"가 되는가, 아니면 남의 분야 목록으로 느껴지는가</p>
<div class="tiles" id="v3t"></div>
<div class="res"><span class="n" id="v3n"></span><div id="v3ex"></div></div>
</section>

<section id="v4">
<h2><b>V4</b>논문으로 분야 정의 — "내 논문과 비슷한 것들"</h2>
<p class="judge">판단 포인트: 카테고리를 하나도 안 고르고 분야에 도착하는 이 경로가 본인에게 자연스러운가 (실서비스에선 제목 검색·arXiv 링크 입력)</p>
<div id="v4s"></div><div id="v4r"></div>
</section>

<section id="v5">
<h2><b>V5</b>같은 문제와 싸우는 논문 — limitation 문장 검색</h2>
<p class="judge">판단 포인트: "주제"가 아니라 "공격하는 문제"로 묶인 이 집합이 실제로 읽고 싶은 목록인가</p>
<div id="v5q"></div><div id="v5r"></div>
<div class="note">논문이 스스로 밝힌 한계 문장(89% 커버리지)에서 검색 · 실서비스에선 자유 입력</div>
</section>

<script>const M=__MOCK__;</script>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const PP=M.papers; // [taskIds, domIds, methIds, title]
const VOC={k:M.W,d:M.D,u:M.U};
const AXIS={k:0,d:1,u:2};

function count(sel){ // sel = {k:Set,d:Set,u:Set}
  const out=[];
  for(const p of PP){
    let ok=true;
    for(const ax of ['k','d','u']){
      if(!sel[ax].size)continue;
      let hit=false;
      for(const id of p[AXIS[ax]]) if(sel[ax].has(id)){hit=true;break;}
      if(!hit){ok=false;break;}
    }
    if(ok)out.push(p);
  }
  return out;
}
function showRes(nEl,exEl,sel){
  const hits=count(sel);
  const any=sel.k.size||sel.d.size||sel.u.size;
  $(nEl).innerHTML=any?`${hits.length.toLocaleString()}<small>of ${M.total.toLocaleString()} papers · ${(100*hits.length/M.total).toFixed(1)}%</small>`
                      :`<small>슬롯을 채우면 여기서 개수가 셉니다</small>`;
  $(exEl).innerHTML=any?hits.slice(0,3).map(p=>`<div class="ex">${esc(p[3])}</div>`).join('')
    +(hits.length>3?`<div class="ex" style="color:var(--mut)">and ${(hits.length-3).toLocaleString()} more</div>`:''):'';
}

// ---------- V1 ----------
const s1={k:new Set(),d:new Set(),u:new Set()};
let dd=null;
function closeDD(){ if(dd){dd.remove();dd=null;} }
document.addEventListener('click',e=>{ if(dd&&!dd.contains(e.target)&&!e.target.closest('.slot'))closeDD(); });
function slotLabel(ax){
  const def={k:'anything',d:'any field',u:'any method'}[ax];
  if(!s1[ax].size)return def;
  const id=[...s1[ax]][0];
  const row=VOC[ax].find(r=>r[0]===id);
  return row?row[1]:def;
}
function paintV1(){
  document.querySelectorAll('#v1 .slot').forEach(b=>{
    const ax=b.dataset.s;
    b.classList.toggle('empty',!s1[ax].size);
    b.innerHTML=esc(slotLabel(ax))+(s1[ax].size?'<span class="x">×</span>':'');
  });
  showRes('#v1n','#v1ex',s1);
}
document.querySelectorAll('#v1 .slot').forEach(b=>b.onclick=e=>{
  const ax=b.dataset.s;
  if(s1[ax].size){ s1[ax].clear(); paintV1(); closeDD(); return; }
  closeDD();
  dd=document.createElement('div'); dd.className='dd';
  // counts conditioned on the OTHER slots — the menu narrows as the sentence fills
  const others={...s1,[ax]:new Set()};
  const base=count(others);
  const cnt=new Map();
  for(const p of base) for(const id of p[AXIS[ax]]) cnt.set(id,(cnt.get(id)||0)+1);
  const rows=VOC[ax].map(([id,l])=>[id,l,cnt.get(id)||0]).filter(r=>r[2]>0)
                    .sort((a,b)=>b[2]-a[2]);
  dd.innerHTML='<input placeholder="filter…">'+rows.slice(0,200).map(([id,l,c])=>
    `<button class="op" data-id="${id}"><span>${esc(l)}</span><b>${c}</b></button>`).join('');
  b.after(dd);
  const inp=dd.querySelector('input'); inp.focus();
  inp.oninput=()=>{const q=inp.value.toLowerCase();
    dd.querySelectorAll('.op').forEach(o=>o.hidden=!o.textContent.toLowerCase().includes(q));};
  dd.querySelectorAll('.op').forEach(o=>o.onclick=()=>{
    s1[ax].clear(); s1[ax].add(+o.dataset.id); paintV1(); closeDD();});
  e.stopPropagation();
});
paintV1();

// ---------- V2 ----------
const s2={k:new Set(),d:new Set(),u:new Set()};
const ALL=[];
for(const ax of ['k','d','u']) for(const [id,l,c] of VOC[ax]) ALL.push({ax,id,l,c});
const ROLE={k:['topic','what it studies'],d:['field','where it is applied'],u:['builds on','used as a tool']};
$('#v2q').oninput=()=>{
  const q=$('#v2q').value.trim().toLowerCase();
  const box=$('#v2s');
  if(q.length<2){box.hidden=true;return;}
  const hits=ALL.filter(x=>x.l.toLowerCase().includes(q)).sort((a,b)=>b.c-a.c).slice(0,9);
  box.innerHTML=hits.map(x=>{
    const amb=M.AMB[x.l.toLowerCase()];
    const cls=x.ax==='d'?'d':x.ax==='u'?'u':'';
    return `<button class="sg" data-ax="${x.ax}" data-id="${x.id}">${esc(x.l)}`+
      `<span class="role ${cls}">${ROLE[x.ax][0]}</span><b>${x.c}</b></button>`;
  }).join('')||'<div class="sg">no match</div>';
  box.hidden=false;
  box.querySelectorAll('.sg[data-id]').forEach(el=>el.onclick=()=>{
    s2[el.dataset.ax].add(+el.dataset.id); box.hidden=true; $('#v2q').value=''; paintV2();});
};
function paintV2(){
  $('#v2c').innerHTML=['k','d','u'].flatMap(ax=>[...s2[ax]].map(id=>{
    const l=(VOC[ax].find(r=>r[0]===id)||[])[1];
    const cls=ax==='d'?'d':ax==='u'?'u':'';
    return `<span class="chip" data-ax="${ax}" data-id="${id}">${esc(l)}`+
      `<span class="role ${cls}">${ROLE[ax][0]}</span>×</span>`;})).join('');
  $('#v2c').querySelectorAll('.chip').forEach(el=>el.onclick=()=>{
    s2[el.dataset.ax].delete(+el.dataset.id); paintV2();});
  showRes('#v2n','#v2ex',s2);
}
paintV2();

// ---------- V3 ----------
$('#v3t').innerHTML=M.PAIRS.map((p,i)=>{
  const kinds={'td':['topic','field'],'tu':['topic','builds on'],'du':['field','builds on']};
  return `<button class="tile" data-i="${i}"><span class="pk">${esc(p.la)}<em>×</em>${esc(p.lb)}</span>`+
    `<b>${p.n} papers · ${kinds[p.kind][0]} × ${kinds[p.kind][1]}</b></button>`;
}).join('');
$('#v3t').querySelectorAll('.tile').forEach(el=>el.onclick=()=>{
  const p=M.PAIRS[+el.dataset.i];
  const sel={k:new Set(),d:new Set(),u:new Set()};
  if(p.kind==='td'){sel.k.add(p.a);sel.d.add(p.b);}
  else if(p.kind==='tu'){sel.k.add(p.a);sel.u.add(p.b);}
  else {sel.d.add(p.a);sel.u.add(p.b);}
  showRes('#v3n','#v3ex',sel);
});

// ---------- V4 ----------
$('#v4s').innerHTML=M.SEEDS.map((sd,i)=>
  `<button class="seed" data-i="${i}">${esc(sd.t)}</button>`).join('');
$('#v4s').querySelectorAll('.seed').forEach(el=>el.onclick=()=>{
  const sd=M.SEEDS[+el.dataset.i];
  $('#v4r').innerHTML=`<div class="res"><span class="n">closest 6<small>임베딩 이웃 — 두 연도에 걸쳐</small></span>`+
    sd.ns.map(n=>`<div class="nb">${esc(n.t)}<i>ICML ${n.y}</i></div>`).join('')+`</div>`;
});

// ---------- V5 ----------
$('#v5q').innerHTML=M.ENEMY.map((e,i)=>
  `<button class="qbtn" data-i="${i}">${esc(e.q)}<b>${e.n}</b></button>`).join('');
$('#v5q').querySelectorAll('.qbtn').forEach(el=>el.onclick=()=>{
  $('#v5q').querySelectorAll('.qbtn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
  const e=M.ENEMY[+el.dataset.i];
  const rx=new RegExp('('+e.q.split(' ')[0].slice(0,9)+'[a-z]*)','ig');
  $('#v5r').innerHTML=`<div class="res"><span class="n">${e.n}<small>papers state this limitation in their own words</small></span></div>`+
    e.ex.map(x=>`<div class="lim"><div class="lt">${esc(x.t)}</div>`+
      `<div class="ls">${esc(x.L).replace(rx,'<mark>$1</mark>')}…</div></div>`).join('');
});
</script></div></body></html>"""

out = Path("reports/mock.html")
out.write_text(HTML.replace("__MOCK__", blob), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
