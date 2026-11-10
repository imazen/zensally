// Read-only CDP checks against a caller-owned Chromium and review gallery.
import {writeFile} from 'node:fs/promises';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));
let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result);}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params);});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value;}
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('section').length>0"))break;await new Promise(r=>setTimeout(r,100));}
const result=await evaluate(`(async()=>{
 const sections=[...document.querySelectorAll('section')],failures=[];
 for(const section of sections){
  for(const key of ['source','aspect','mode'])document.getElementById(key).value=section.dataset[key];
  document.getElementById('mode').dispatchEvent(new Event('change'));
  const visible=sections.filter(s=>!s.hidden);
  if(visible.length!==1||visible[0]!==section)failures.push('selector '+JSON.stringify(section.dataset));
  const svg=section.querySelector('svg'),box=svg.viewBox.baseVal;
  for(const r of svg.querySelectorAll('rect'))if(r.x.baseVal.value<0||r.y.baseVal.value<0||r.x.baseVal.value+r.width.baseVal.value>box.width||r.y.baseVal.value+r.height.baseVal.value>box.height)failures.push('overlay out of bounds');
 }
 document.getElementById('previous').click();
 if(sections.findIndex(s=>!s.hidden)!==sections.length-2)failures.push('previous navigation');
 document.getElementById('next').click();
 if(sections.findIndex(s=>!s.hidden)!==sections.length-1)failures.push('next navigation');
 document.getElementById('accepted').click();
 if(sections.some(s=>s.dataset.selected==='true')&&sections.find(s=>!s.hidden).dataset.selected!=='true')failures.push('accepted navigation');
 const urls=[...new Set([...document.querySelectorAll('img')].map(i=>i.src).concat([...document.querySelectorAll('svg image')].map(i=>new URL(i.getAttribute('href'),location.href).href)))];
 for(let i=0;i<urls.length;i+=8)await Promise.all(urls.slice(i,i+8).map(async url=>{const image=new Image();image.src=url;try{await image.decode();if(!image.naturalWidth||!image.naturalHeight)throw Error('zero size');}catch(e){failures.push(url+': '+e.message);}}));
 return {sections:sections.length,unique_images_decoded:urls.length,failures};
})()`);
result.exceptions=exceptions;
await writeFile(output,JSON.stringify(result,null,2)+'\n');
ws.close();
if(result.sections!==Number(expected)||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
console.log(JSON.stringify(result));
