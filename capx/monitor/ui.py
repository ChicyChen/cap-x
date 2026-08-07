"""Single-file dashboard for the real-robot monitor."""

DASHBOARD_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>CaP-X real-robot monitor</title>
<style>
:root{color-scheme:dark;--bg:#15171c;--panel:#1c1f26;--line:#2c313b;--fg:#dfe3ea;
--dim:#8b94a3;--acc:#6cb6ff;--ok:#4ec9b0;--warn:#e2c08d;--err:#f48771;--tool:#c586c0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:9;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
padding:9px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
h1{font-size:13px;margin:0;font-weight:700;color:var(--acc);letter-spacing:.05em;text-transform:uppercase}
.pill{font-size:11.5px;padding:2px 8px;border-radius:10px;background:#252a33;color:var(--dim)}
.pill b{color:var(--fg)}
#live{margin-left:auto;font-size:12px;color:var(--ok)}
#live.off{color:var(--err)}
main{display:flex;height:calc(100vh - 48px)}
#side{width:270px;border-right:1px solid var(--line);overflow:auto;flex:none}
#feed{flex:1;overflow:auto;padding:10px 14px}
.ep{padding:9px 12px;border-bottom:1px solid var(--line);cursor:pointer}
.ep:hover{background:#20242c}
.ep.sel{background:#232936;border-left:3px solid var(--acc)}
.ep .id{font-weight:700}
.ep .task{color:var(--dim);font-size:12px;margin-top:2px;
overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ep .st{font-size:11px;color:var(--dim);margin-top:4px}
.ev{border-left:3px solid var(--line);padding:5px 10px;margin:5px 0;background:#191d24;border-radius:0 5px 5px 0}
.ev .h{display:flex;gap:8px;align-items:baseline;font-size:11.5px;color:var(--dim)}
.ev .k{font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.ev .t{white-space:pre-wrap;margin-top:3px}
.ev.tool{border-left-color:var(--tool)} .ev.tool .k{color:var(--tool)}
.ev.error{border-left-color:var(--err);background:#2a1e1e} .ev.error .k{color:var(--err)}
.ev.task{border-left-color:var(--acc);background:#1b2430} .ev.task .k{color:var(--acc)}
.ev.code{border-left-color:var(--warn)} .ev.code .k{color:var(--warn)}
.ev.episode_start,.ev.episode_end{border-left-color:var(--ok);background:#1b2724}
.ev.episode_start .k,.ev.episode_end .k{color:var(--ok)}
.ev.motion{border-left-color:#7aa2f7} .ev.motion .k{color:#7aa2f7}
pre{margin:4px 0 0;padding:8px;background:#12151a;border:1px solid var(--line);
border-radius:5px;overflow:auto;font:12px/1.55 "SFMono-Regular",Consolas,monospace;max-height:340px}
img{max-width:260px;border:1px solid var(--line);border-radius:4px;margin:5px 5px 0 0;vertical-align:top;cursor:zoom-in}
img.zoom{max-width:92vw;position:fixed;inset:3vh 4vw auto;z-index:50;box-shadow:0 8px 40px #000}
.muted{color:var(--dim)}
label{font-size:12px;color:var(--dim)}
select,button{font:inherit;font-size:12px;padding:3px 8px;border-radius:5px;
border:1px solid #3a414e;background:#252a33;color:var(--fg);cursor:pointer}
</style></head><body>
<header>
  <h1>CaP-X real robot</h1>
  <span class="pill" id="p-ep">episode <b>—</b></span>
  <span class="pill" id="p-model">model <b>—</b></span>
  <span class="pill" id="p-endpoint">endpoint <b>—</b></span>
  <label>show <select id="filter">
    <option value="all">everything</option>
    <option value="tool">tools only</option>
    <option value="code">code only</option>
    <option value="error">errors only</option>
  </select></label>
  <button onclick="clearFeed()">clear</button>
  <span id="live">connecting…</span>
</header>
<main>
  <div id="side"></div>
  <div id="feed"><div class="muted">waiting for events…</div></div>
</main>
<script>
const feed=document.getElementById('feed'), side=document.getElementById('side');
let events=[], episodes=[], sel=null, filter='all';
document.getElementById('filter').onchange=e=>{filter=e.target.value;render()};
function clearFeed(){events=[];render()}
function fmt(ts){const d=new Date(ts*1000);return d.toTimeString().slice(0,8)}
function keep(e){
  if(sel!==null && String(e.episode)!==String(sel)) return false;
  if(filter==='all') return true;
  if(filter==='tool') return e.kind==='tool';
  if(filter==='code') return e.kind==='code';
  if(filter==='error') return e.kind==='error';
  return true;
}
function render(){
  const rows=events.filter(keep).slice(-500);
  feed.innerHTML = rows.length? rows.map(e=>{
    const imgs=(e.images||[]).map(s=>`<img src="data:image/png;base64,${s}" onclick="this.classList.toggle('zoom')">`).join('');
    const code=e.data&&e.data.code?`<pre>${esc(e.data.code)}</pre>`:'';
    const extra=e.data&&Object.keys(e.data).length&&e.kind!=='code'
      ? `<div class="muted" style="font-size:11.5px">${esc(JSON.stringify(e.data))}</div>`:'';
    const stale = e.data && e.data.stale;
    const annot = e.data && e.data.annotated === true;
    const rawin = e.data && e.data.annotated === false && (e.images||[]).length;
    return `<div class="ev ${e.kind}" ${stale?'style="opacity:.45"':''}><div class="h"><span class="k">${e.kind}</span>
      ${stale?'<span style="color:var(--warn)">↩ previous episode still unwinding</span>':''}
      ${annot?'<span style="color:var(--ok)">✎ annotated result</span>':''}
      ${rawin?'<span class="muted">input frame</span>':''}
      <span>${fmt(e.ts)}</span>${e.episode!=null?`<span>ep ${esc(String(e.episode))}</span>`:''}</div>
      ${e.text?`<div class="t">${esc(e.text)}</div>`:''}${code}${extra}${imgs}</div>`;
  }).join('') : '<div class="muted">no events match this filter</div>';
  feed.scrollTop=feed.scrollHeight;
  side.innerHTML = `<div class="ep ${sel===null?'sel':''}" onclick="pick(null)">
      <div class="id">All episodes</div><div class="st">${events.length} events</div></div>` +
    episodes.slice().reverse().map(ep=>{
      // "running" is only credible while events keep arriving. Without this a
      // stopped episode ticks up forever, because the driver sends no
      // end-of-episode message (see _announce_episode_end).
      const last = ep.last_event || ep.started;
      const idle = Date.now()/1000 - last;
      const dur = ep.ended
        ? Math.round(ep.ended-ep.started)+'s'
        : (idle > 12
            ? Math.round(last-ep.started)+'s <span style="color:var(--warn)">idle '
              + Math.round(idle) + 's</span>'
            : Math.round(Date.now()/1000-ep.started)+'s ⏵');
      const tools=Object.entries(ep.tools||{}).map(([k,v])=>k+'×'+v).join(', ');
      return `<div class="ep ${String(sel)===String(ep.episode)?'sel':''}" onclick="pick(${JSON.stringify(ep.episode)})">
        <div class="id">Episode ${esc(String(ep.episode))}</div>
        <div class="task">${esc(ep.task||'(no task yet)')}</div>
        <div class="st">${dur} · ${ep.model_calls} llm · ${ep.code_blocks} code ·
          ${ep.waypoints} wp${ep.errors?` · <span style="color:var(--err)">${ep.errors} err</span>`:''}</div>
        ${tools?`<div class="st">${esc(tools)}</div>`:''}</div>`;
    }).join('');
}
function pick(e){sel=e;render()}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function apply(e){
  events.push(e);
  if(e.kind==='session'){
    if(e.data.model) document.querySelector('#p-model b').textContent=e.data.model;
    if(e.data.endpoint) document.querySelector('#p-endpoint b').textContent=e.data.endpoint;
  }
  if(e.kind==='episode_start'){
    document.querySelector('#p-ep b').textContent=e.episode;
    if(!episodes.find(x=>String(x.episode)===String(e.episode)))
      episodes.push({episode:e.episode,started:e.ts,ended:null,task:null,tools:{},
                     model_calls:0,code_blocks:0,errors:0,waypoints:0,chunks:0});
  }
  const ep=episodes.find(x=>String(x.episode)===String(e.episode));
  if(ep && !(e.data&&e.data.stale)){
    if(e.kind==='episode_end') ep.ended=e.ts;
    else if(e.kind==='task') ep.task=e.text;
    else if(e.kind==='tool'){const n=(e.data&&e.data.tool)||'?';ep.tools[n]=(ep.tools[n]||0)+1}
    else if(e.kind==='model_query'&&e.data&&e.data.phase==='end') ep.model_calls++;
    else if(e.kind==='code') ep.code_blocks++;
    else if(e.kind==='error') ep.errors++;
    else if(e.kind==='motion') ep.waypoints+=(e.data&&e.data.waypoints)||0;
  }
}
fetch('/api/snapshot').then(r=>r.json()).then(s=>{
  episodes=s.episodes||[]; (s.events||[]).forEach(apply);
  if(s.current_episode!=null) document.querySelector('#p-ep b').textContent=s.current_episode;
  render(); connect();
}).catch(()=>connect());
function connect(){
  const es=new EventSource('/api/stream');
  const live=document.getElementById('live');
  es.onopen=()=>{live.textContent='live';live.className=''};
  es.onerror=()=>{live.textContent='reconnecting…';live.className='off'};
  es.onmessage=m=>{try{apply(JSON.parse(m.data));render()}catch(_){}};
}
setInterval(()=>{if(sel===null)render()},5000);
</script></body></html>
'''
