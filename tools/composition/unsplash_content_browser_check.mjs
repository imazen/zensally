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
await call('Emulation.setDeviceMetricsOverride',{width:1680,height:900,deviceScaleFactor:1,mobile:false});
const screenshots=join(dirname(output),'screenshots');await mkdir(screenshots);
const result={sources:Number(expected),sheets:[],decoded:0,screenshot_bytes:0,exceptions};
for(let index=0;index<Math.ceil(Number(expected)/10);index++){
 const row=await evaluate(`(async()=>{sheet.value=${index};render();const rows=data.sources.filter(r=>r.sheet===${index}),articles=[...tiles.querySelectorAll('article')],failures=[];if(rows.length!==articles.length)failures.push('tile count');for(let j=0;j<rows.length;j++){const image=articles[j].querySelector('img');await image.decode();if(image.naturalWidth!==rows[j].preview_size[0]||image.naturalHeight!==rows[j].preview_size[1])failures.push('source dimensions');if(getComputedStyle(image).objectFit!=='contain')failures.push('cropped or stretched source');if(!(data.dataset_kind==='owned-screening'?['screening']:['fit','dev']).includes(rows[j].role))failures.push('forbidden role');if(articles[j].dataset.id!==rows[j].id||articles[j].querySelector('a').href!==new URL(rows[j].source.source_url).href)failures.push('source/link identity')}return{sheet:${index},ids:rows.map(r=>r.id),decoded:articles.length,failures}})()`);
 if(row.failures.length)throw Error(JSON.stringify(row));result.sheets.push(row);result.decoded+=row.decoded;
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false}),bytes=Buffer.from(shot.data,'base64');result.screenshot_bytes+=bytes.length;
 if(result.screenshot_bytes>24*1024*1024)throw Error('registered screenshot budget exceeded');
 await writeFile(join(screenshots,'sheet-'+String(index).padStart(2,'0')+'.png'),bytes);
}
if(result.decoded!==Number(expected)||new Set(result.sheets.flatMap(s=>s.ids)).size!==Number(expected)||exceptions.length)throw Error(JSON.stringify(result));
await writeFile(output,JSON.stringify(result,null,2)+'\n');ws.close();console.log(JSON.stringify({sources:result.sources,decoded:result.decoded,sheets:result.sheets.length,screenshot_bytes:result.screenshot_bytes,exceptions}));
