import {writeFile} from 'node:fs/promises';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result);}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params);});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value;}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('article').length===8"))break;await new Promise(r=>setTimeout(r,100));}
const result=await evaluate(`(async()=>{
 const failures=[],counts={};
 for(const key of ['pose','gaze','scan']){
  for(const input of document.querySelectorAll('input'))input.checked=input.id===key;
  document.querySelector('#'+key).dispatchEvent(new Event('change'));
  counts[key]={circles:document.querySelectorAll('circle').length,lines:document.querySelectorAll('line').length};
  if(!counts[key].circles||!counts[key].lines)failures.push('missing '+key+' vectors');
  for(const el of document.querySelectorAll('circle,line'))for(const a of [...el.attributes])if(['cx','cy','x1','y1','x2','y2'].includes(a.name)&&!Number.isFinite(Number(a.value)))failures.push('nonfinite vector');
 }
 const images=[...document.querySelectorAll('svg image')];
 for(const el of images){const image=new Image();image.src=el.getAttribute('href');try{await image.decode();if(!image.naturalWidth)failures.push('empty image')}catch(e){failures.push('image decode: '+image.src)}}
 return {sources:document.querySelectorAll('article').length,images:images.length,counts,failures};
})()`);
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();
if(result.sources!==Number(expected)||result.images!==Number(expected)||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));console.log(JSON.stringify(result));
