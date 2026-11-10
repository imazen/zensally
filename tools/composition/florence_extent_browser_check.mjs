import {writeFile} from 'node:fs/promises';
const[endpoint,url,expected,output]=process.argv.slice(2);
const pages=await(await fetch(endpoint+'/json/list')).json(),ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;
const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate(`typeof data!=='undefined'&&data.sources.length===${Number(expected)}`))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{let decoded=0,entities=0,predictions=0,checks=0;const failures=[];for(const row of data.sources){const image=new Image();image.src=row.image;await image.decode();if(image.naturalWidth!==row.size[0]||image.naturalHeight!==row.size[1])failures.push('dimensions '+row.id);decoded++;entities+=row.entities.length;for(const key of Object.keys(row.models)){source.value=row.id;model.value=key;render();const svg=view.querySelector('svg');if(svg.getAttribute('viewBox')!==[0,0,...row.size].join(' ')||svg.getAttribute('preserveAspectRatio')!=='xMidYMid meet')failures.push('geometry');if(svg.querySelectorAll('[data-kind=entity]').length!==row.entities.length)failures.push('entity count');const n=svg.querySelectorAll('[data-kind=prediction]').length;if(n!==(row.role==='dev'?1:0))failures.push('prediction scope');predictions+=n;checks++}}source.value=data.sources.find(s=>s.role==='dev').id;model.value='focal-31';render();return{sources:data.sources.length,decoded,entities,predictions,checks,failures}})()`);
result.exceptions=exceptions;
if(result.sources!==56||result.decoded!==56||result.entities!==101||result.predictions!==48||result.checks!==336||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
await writeFile(output,JSON.stringify(result,null,2)+'\n');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1500,deviceScaleFactor:1,mobile:false});
const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
await writeFile(output.replace(/\.json$/,'.png'),Buffer.from(shot.data,'base64'));
ws.close();console.log(JSON.stringify(result));
