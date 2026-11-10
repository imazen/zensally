import {writeFile} from 'node:fs/promises';
const[endpoint,url,expected,output]=process.argv.slice(2);
const pages=await(await fetch(endpoint+'/json/list')).json();
const ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;
const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate("document.querySelectorAll('article').length===8"))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{const failures=[];let decoded=0;for(const image of document.querySelectorAll('img')){await image.decode();if(!image.naturalWidth||getComputedStyle(image).objectFit!=='contain')failures.push('image empty or stretched');decoded++}for(const row of data){const article=[...document.querySelectorAll('article')].find(e=>e.dataset.task===[row.id,row.ratio,row.mode].join('|'));if(!article||article.querySelectorAll('img').length!==3)failures.push('missing task');for(const item of row.images)if(item.action==='crop'){const image=[...article.querySelectorAll('img')].find(e=>e.getAttribute('src')===item.file);const[p,q]=row.ratio.split(':').map(Number);if(image.naturalWidth*q!==image.naturalHeight*p)failures.push('wrong crop ratio')}}return{tasks:data.length,decoded,failures}})()`);
result.exceptions=exceptions;if(result.tasks!==Number(expected)||result.decoded!==24||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();console.log(JSON.stringify(result));
