import {mkdir,writeFile} from 'node:fs/promises';
import {dirname,join} from 'node:path';
const[endpoint,url,expected,output]=process.argv.slice(2);
const pages=await(await fetch(endpoint+'/json/list')).json(),ws=new WebSocket(pages.find(p=>p.type==='page').webSocketDebuggerUrl);
await new Promise(r=>ws.addEventListener('open',r,{once:true}));let serial=0;
const pending=new Map(),exceptions=[];
ws.addEventListener('message',event=>{const v=JSON.parse(event.data);if(v.id){const p=pending.get(v.id);pending.delete(v.id);v.error?p.reject(v.error):p.resolve(v.result)}else if(v.method==='Runtime.exceptionThrown')exceptions.push(v.params)});
function call(method,params={}){return new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))})}
async function evaluate(expression){const v=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(v.exceptionDetails)throw Error(JSON.stringify(v.exceptionDetails));return v.result.value}
await call('Runtime.enable');await call('Page.enable');await call('Page.navigate',{url});
for(let i=0;i<100;i++){if(await evaluate(`typeof data!=='undefined'&&data.sources.length===${Number(expected)}`))break;await new Promise(r=>setTimeout(r,100))}
await call('Emulation.setDeviceMetricsOverride',{width:1680,height:2000,deviceScaleFactor:1,mobile:false});
const result=await evaluate(`(async()=>{const failures=[];let decoded=0,checks=0;for(const row of data.sources){if(row.role!=='dev')failures.push('forbidden role');const image=new Image();image.src=row.image;await image.decode();if(!image.naturalWidth)failures.push('empty source');decoded++}for(const value of data.ratios)for(const amount of ['minimal','maximal']){ratio.value=value;mode.value=amount;render();const articles=[...document.querySelectorAll('article')];if(articles.length!==8)failures.push('missing source');for(let i=0;i<articles.length;i++){const row=data.sources[i],svgs=[...articles[i].querySelectorAll('svg')],task=row.tasks.find(t=>t.ratio===value&&t.mode===amount);if(svgs.length!==5)failures.push('missing comparison');if(articles[i].dataset.id!==row.id||articles[i].querySelector('a').href!==new URL(row.source_url).href)failures.push('source/link differs');for(let j=1;j<5;j++){const svg=svgs[j],image=svg.querySelector('image'),box=task.boxes[['teacher','baseline','candidate','geometry'][j-1]],rect=svg.querySelector('clipPath rect');if(!rect||!image.getAttribute('clip-path'))failures.push('unclipped crop');if(svg.getAttribute('preserveAspectRatio')!=='xMidYMid meet')failures.push('stretched crop');if(svg.getAttribute('viewBox')!==[box[0],box[1],box[2]-box[0],box[3]-box[1]].join(' '))failures.push('wrong crop');checks++}}}return{sources:data.sources.length,decoded,checks,failures}})()`);
if(result.sources!==Number(expected)||result.decoded!==8||result.checks!==704||result.failures.length||exceptions.length)throw Error(JSON.stringify({...result,exceptions}));
const screenshots=join(dirname(output),'screenshots');await mkdir(screenshots);result.screenshots=[];result.screenshot_bytes=0;
for(const value of ['3:1','1:3','1:1','6:1','1:6'])for(const amount of ['minimal','maximal']){
 await evaluate(`ratio.value=${JSON.stringify(value)};mode.value=${JSON.stringify(amount)};render()`);
 const height=await evaluate('Math.ceil(document.documentElement.scrollHeight)'),shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width:1680,height,scale:1}}),bytes=Buffer.from(shot.data,'base64');
 result.screenshot_bytes+=bytes.length;if(result.screenshot_bytes>await evaluate('data.screenshot_budget_bytes'))throw Error('review screenshot budget exceeded');
 const name=value.replace(':','x')+'-'+amount+'.png';await writeFile(join(screenshots,name),bytes);result.screenshots.push(name);
}
result.exceptions=exceptions;await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();console.log(JSON.stringify(result));
