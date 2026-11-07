// Use a task-owned Chromium CDP endpoint; protocol is read from that browser.
import fs from 'node:fs';
const [url, output, endpoint='http://127.0.0.1:3993', expectedMethods='8'] = process.argv.slice(2);
if (!url || !output) throw Error('usage: browser_check.mjs URL OUTPUT [CDP_ENDPOINT]');
fs.mkdirSync(output,{recursive:true});
const pages=await (await fetch(`${endpoint}/json/list`)).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
let serial=0;const pending=new Map();const failures=[];
ws.onmessage=event=>{const msg=JSON.parse(event.data);if(msg.id){const q=pending.get(msg.id);pending.delete(msg.id);if(!q)return;if(msg.error)q.reject(Error(JSON.stringify(msg.error)));else q.resolve(msg.result);}else if(msg.method==='Runtime.exceptionThrown')failures.push(msg.params.exceptionDetails);};
function send(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
async function evaluate(expression,awaitPromise=false){const r=await send('Runtime.evaluate',{expression,awaitPromise,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
await send('Runtime.enable');await send('Page.enable');await send('Page.navigate',{url});
for(let attempt=0;attempt<100;attempt++){
 if(await evaluate(`document.querySelectorAll(".card").length===${+expectedMethods}`))break;
 await new Promise(resolve=>setTimeout(resolve,100));
}
const count=await evaluate('document.getElementById("case").options.length');if(count!==29)throw Error(`Wrong case count ${count}`);
async function ready(){return evaluate(`(async()=>{await Promise.all([...document.images].map(i=>i.decode()));return {cards:document.querySelectorAll('.card').length,images:document.images.length,broken:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),details:document.getElementById('details').textContent}})()`,true);}
const checks=[];
for(let index=0;index<count;index++){
 await evaluate(`document.getElementById('case').value=${index};document.getElementById('case').dispatchEvent(new Event('change'))`);
 const state=await ready();if(state.cards!==+expectedMethods||state.broken.length)throw Error(JSON.stringify(state));checks.push(state);
 if(index===0||index===count-1){const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});fs.writeFileSync(`${output}/${index===0?'natural':'extreme'}-review.png`,Buffer.from(shot.data,'base64'));}
}
await evaluate(`document.getElementById('mode').value='maximal';document.getElementById('aspect').value='9:16';document.getElementById('aspect').dispatchEvent(new Event('change'));document.getElementById('opacity').value=0.2;document.getElementById('opacity').dispatchEvent(new Event('input'))`);
checks.push(await ready());if(failures.length)throw Error(JSON.stringify(failures));
const result={cases_checked:count,methods_checked:+expectedMethods,control_checks:['case selection','9:16 maximal crop','heatmap opacity'],javascript_errors:failures,checks};fs.writeFileSync(`${output}/browser-check.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({cases_checked:count,javascript_errors:failures.length,screenshots:2}));
ws.send(JSON.stringify({id:++serial,method:'Browser.close',params:{}}));
setTimeout(()=>process.exit(0),200);
