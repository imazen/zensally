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
for(let i=0;i<100;i++){if(await evaluate(`document.querySelectorAll('article').length===${Number(expected)}`))break;await new Promise(r=>setTimeout(r,100))}
const result=await evaluate(`(async()=>{const failures=[];let decoded=0,checks=0;for(const row of data.sources){const image=new Image();image.src=row.image;await image.decode();if(!image.naturalWidth)failures.push('empty source');decoded++}for(const value of ['3:1','1:3','6:1','1:6','1:1'])for(const amount of ['minimal','maximal']){ratio.value=value;mode.value=amount;render();const articles=[...document.querySelectorAll('article')];if(articles.length!==6)failures.push('missing source');for(let i=0;i<articles.length;i++){const svgs=[...articles[i].querySelectorAll('svg')],task=data.sources[i].tasks.find(t=>t.ratio===value&&t.mode===amount);if(svgs.length!==4)failures.push('missing comparison');for(let j=1;j<4;j++){const svg=svgs[j],image=svg.querySelector('image'),box=task.boxes[['teacher','student','geometry'][j-1]],rect=svg.querySelector('clipPath rect');if(!rect||!image.getAttribute('clip-path'))failures.push('unclipped crop');if(svg.getAttribute('preserveAspectRatio')!=='xMidYMid meet')failures.push('stretched crop');if(svg.getAttribute('viewBox')!==[box[0],box[1],box[2]-box[0],box[3]-box[1]].join(' '))failures.push('wrong crop coordinates');checks++}}}ratio.value='3:1';mode.value='minimal';render();return{sources:data.sources.length,decoded,checks,failures}})()`);
result.exceptions=exceptions;
if(result.sources!==Number(expected)||result.decoded!==6||result.checks!==180||result.failures.length||exceptions.length)throw Error(JSON.stringify(result));
await writeFile(output,JSON.stringify(result,null,2)+'\n');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1600,deviceScaleFactor:1,mobile:false});
const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
await writeFile(output.replace(/\.json$/,'.png'),Buffer.from(shot.data,'base64'));
ws.close();console.log(JSON.stringify(result));
