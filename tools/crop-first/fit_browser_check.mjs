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
 const source=document.querySelector('#source'),ratio=document.querySelector('#ratio'),mode=document.querySelector('#mode'),failures=[],counts=[];
 const positions=[0,Math.floor(source.options.length/3),Math.floor(source.options.length*2/3),source.options.length-1];
 for(const position of positions){source.value=position;source.dispatchEvent(new Event('change'));for(const r of ['3:1','1:3','6:1','1:6']){ratio.value=r;ratio.dispatchEvent(new Event('change'));mode.value='all';mode.dispatchEvent(new Event('change'));const images=[...document.querySelectorAll('img')];for(const image of images){image.loading='eager';try{await image.decode()}catch(e){failures.push('image decode: '+image.src)}}await new Promise(requestAnimationFrame);if(document.querySelectorAll('section').length!==2)failures.push('both modes missing');for(const image of images){const b=image.getBoundingClientRect();if(Math.abs(b.width/b.height-image.naturalWidth/image.naturalHeight)>.01)failures.push('stretched image');}counts.push({source:source.options[position].text,ratio:r,crops:images.length-1});}}
 return {sources:source.options.length,ratios:ratio.options.length-1,checks:counts,failures,error:document.querySelector('#error').textContent};
})()`);
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();
if(result.sources!==Number(expected)||result.ratios!==11||result.checks.length!==16||result.failures.length||result.error||exceptions.length)throw Error(JSON.stringify(result));console.log(JSON.stringify(result));
