import {writeFile} from 'node:fs/promises';
const [endpoint,url,expected,output]=process.argv.slice(2);
const pages=await (await fetch(endpoint+'/json/list')).json();const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result);}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params);});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value;}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('section').length>0"))break;await new Promise(r=>setTimeout(r,100));}
const result=await evaluate(`(async()=>{
 const sections=[...document.querySelectorAll('section')],failures=[];
 for(const details of document.querySelectorAll('details'))details.open=true;
 for(const svg of document.querySelectorAll('svg')){const b=svg.viewBox.baseVal;for(const r of svg.querySelectorAll('rect'))if(r.x.baseVal.value<0||r.y.baseVal.value<0||r.x.baseVal.value+r.width.baseVal.value>b.width+.0001||r.y.baseVal.value+r.height.baseVal.value>b.height+.0001)failures.push('out of bounds overlay');}
 const urls=[...new Set([...document.querySelectorAll('img')].map(i=>i.src).concat([...document.querySelectorAll('svg image')].map(i=>new URL(i.getAttribute('href'),location.href).href)))];
 for(let i=0;i<urls.length;i+=8)await Promise.all(urls.slice(i,i+8).map(async url=>{const image=new Image();image.src=url;try{await image.decode();}catch(e){failures.push(url+': '+e.message);}}));
 return {sections:sections.length,requests:document.querySelectorAll('details').length,crop_images:document.querySelectorAll('img').length,unique_images_decoded:urls.length,failures};
})()`);
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();
if(result.sections!==Number(expected)||result.requests!==56||result.crop_images!==44||result.unique_images_decoded!==48||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));console.log(JSON.stringify(result));
