'use strict';
const $=id=>document.getElementById(id);
const orders={upper:[17,16,15,14,13,12,11,21,22,23,24,25,26,27],lower:[47,46,45,44,43,42,41,31,32,33,34,35,36,37]};
const q=Object.fromEntries(Object.entries(orders).map(([a,labels])=>[a,Object.fromEntries(labels.map(k=>[k,Array(10).fill(0)]))]));
const jobs={upper:{revision:0,token:0,busy:false,timer:null},lower:{revision:0,token:0,busy:false,timer:null}};
const meshData={},gpuMeshes={};
let selectedArch='upper',selectedTooth=11,ready=false,metadata=null,opening=6,visibility='both',wire=false,color=true;
let yaw=.45,elevation=.10,zoom=1,pan=[0,.10],lastPointer=null;
const canvas=$('viewer'),gl=canvas.getContext('webgl',{antialias:true,alpha:true,preserveDrawingBuffer:true});
let program;

function fail(message){$('decode-status').textContent=message;$('decode-status').classList.add('error');}
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function initGL(){
 if(!gl)throw Error('当前浏览器不支持 WebGL，请使用启用硬件加速的 Chrome 或 Edge。');
 program=gl.createProgram();
 gl.attachShader(program,shader(gl.VERTEX_SHADER,`
 attribute vec3 position;attribute vec3 normal;attribute float gum;attribute float label;
 uniform mat3 camera;uniform vec3 offset;uniform vec3 center;uniform vec2 scale;uniform vec2 pan;
 uniform float angle;uniform float hinge;uniform float lower;
 varying vec3 vNormal;varying float vGum;varying float vLabel;
 void main(){vec3 p=position+offset;vec3 n=normal;
 if(lower>.5){float c=cos(angle),s=sin(angle),y=p.y-hinge,z=p.z;p.y=c*y-s*z+hinge;p.z=s*y+c*z;n=vec3(n.x,c*n.y-s*n.z,s*n.y+c*n.z);}
 vec3 view=camera*(p-center);vNormal=camera*n;vGum=gum;vLabel=label;
 gl_Position=vec4(view.xy*scale+pan,-view.z/220.,1.);}`));
 gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`
 precision mediump float;varying vec3 vNormal;varying float vGum;varying float vLabel;
 uniform float selected;uniform float line;uniform float colored;
 void main(){vec3 n=normalize(vNormal);if(n.z<0.)n=-n;
 float lighting=.36+.58*max(dot(n,normalize(vec3(-.35,.45,.82))),0.)+.06*max(n.z,0.);
 vec3 ivory=vec3(.965,.929,.837),pink=vec3(.78,.53,.54);
 vec3 base=mix(ivory,mix(pink,ivory,smoothstep(.18,.65,1.-vGum)),colored);
 float highlight=1.-step(.4,abs(vLabel-selected));base=mix(base,vec3(.34,.72,.85),highlight*.66);
 gl_FragColor=vec4(mix(base*lighting,vec3(.21,.36,.43),line),1.);}`));
 gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
 gl.enable(gl.DEPTH_TEST);gl.useProgram(program);
}
function expanded(data,edges=false){
 const result=new Float32Array(data.faces.length*(edges?6:3)*8);let at=0;
 for(const f of data.faces){const ids=edges?[f[0],f[1],f[1],f[2],f[2],f[0]]:f;
  for(const i of ids){result.set(data.vertices[i],at);result.set(data.normals[i],at+3);result[at+6]=data.gum[i];result[at+7]=data.labels[i];at+=8;}}
 return result;
}
function buffer(values){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);return b;}
function installMesh(arch,data){
 if(data.arch!==arch||!Array.isArray(data.faces)||!data.faces.length)throw Error('网格响应不完整');
 if(gpuMeshes[arch]){gl.deleteBuffer(gpuMeshes[arch].solid);if(gpuMeshes[arch].edges)gl.deleteBuffer(gpuMeshes[arch].edges);}
 meshData[arch]=data;gpuMeshes[arch]={solid:buffer(expanded(data)),edges:null};draw();updateCount();
}
function bind(b){gl.bindBuffer(gl.ARRAY_BUFFER,b);for(const [name,size,offset] of [['position',3,0],['normal',3,12],['gum',1,24],['label',1,28]]){const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,32,offset);}}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function draw(){
 if(!gl||!program)return;
 const ratio=Math.min(devicePixelRatio,2),w=Math.round(canvas.clientWidth*ratio),h=Math.round(canvas.clientHeight*ratio);
 if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
 gl.viewport(0,0,w,h);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(program);
 const z=[Math.sin(yaw)*Math.cos(elevation),-Math.cos(yaw)*Math.cos(elevation),Math.sin(elevation)],x=[Math.cos(yaw),Math.sin(yaw),0],y=cross(z,x);
 gl.uniformMatrix3fv(gl.getUniformLocation(program,'camera'),false,new Float32Array([x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]]));
 const aspect=w/h,extent=Math.max(65,74*aspect);
 gl.uniform2f(gl.getUniformLocation(program,'scale'),zoom/extent,zoom*aspect/extent);gl.uniform2fv(gl.getUniformLocation(program,'pan'),pan);
 gl.uniform3f(gl.getUniformLocation(program,'center'),0,28,visibility==='upper'?12:visibility==='lower'?-16:0);
 gl.uniform1f(gl.getUniformLocation(program,'colored'),color?1:0);gl.uniform1f(gl.getUniformLocation(program,'angle'),opening*Math.PI/180);
 for(const arch of ['upper','lower']){if(!gpuMeshes[arch]||(visibility!=='both'&&visibility!==arch))continue;
  const data=meshData[arch],gpu=gpuMeshes[arch],p=data.placement;
  gl.uniform3f(gl.getUniformLocation(program,'offset'),p.x,p.y,p.z);gl.uniform1f(gl.getUniformLocation(program,'hinge'),p.hinge_y);
  gl.uniform1f(gl.getUniformLocation(program,'lower'),arch==='lower'?1:0);gl.uniform1f(gl.getUniformLocation(program,'selected'),arch===selectedArch?selectedTooth:-100);
  gl.uniform1f(gl.getUniformLocation(program,'line'),0);bind(gpu.solid);gl.enable(gl.POLYGON_OFFSET_FILL);gl.polygonOffset(1,1);gl.drawArrays(gl.TRIANGLES,0,data.faces.length*3);gl.disable(gl.POLYGON_OFFSET_FILL);
  if(wire){if(!gpu.edges)gpu.edges=buffer(expanded(data,true));bind(gpu.edges);gl.uniform1f(gl.getUniformLocation(program,'line'),1);gl.drawArrays(gl.LINES,0,data.faces.length*6);}
 }
}
function setView(view){
 pan=[0,.10];zoom=1;
 if(view==='front'){yaw=0;elevation=0;}else if(view==='upper'){yaw=0;elevation=-Math.PI/2;setVisibility('upper');}
 else if(view==='lower'){yaw=0;elevation=Math.PI/2;setVisibility('lower');}else{yaw=.45;elevation=.10;}
 document.querySelectorAll('#views button').forEach(b=>b.classList.toggle('active',b.dataset.view===view));draw();
}
function setVisibility(mode){visibility=mode;document.querySelectorAll('#visibility button').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));draw();}
function updateCount(){let triangles=0;for(const a of ['upper','lower'])if(meshData[a])triangles+=meshData[a].faces.length;
 $('mesh-count').textContent=`${metadata?metadata.resolution+'³ · ':''}${triangles.toLocaleString()} 三角面`;
}
function updateTeeth(){for(const a of ['upper','lower'])for(const k of orders[a]){const b=$(`tooth-${k}`);b.classList.toggle('selected',a===selectedArch&&k===selectedTooth);b.classList.toggle('changed',q[a][k].some(v=>v!==0));}}
function selectTooth(a,k){selectedArch=a;selectedTooth=k;$('selected-title').textContent=`${a==='upper'?'上颌':'下颌'} · ${k}`;
 for(let i=0;i<10;i++){$(`q-${i}`).value=q[a][k][i];$(`value-${i}`).value=Number(q[a][k][i]).toFixed(1);}
 if(visibility!=='both'&&visibility!==a)setVisibility('both');updateTeeth();draw();
}
function dirty(a){return !meshData[a]||JSON.stringify(q[a])!==JSON.stringify(meshData[a].q);}
function updateExport(){$('export-mesh').disabled=!ready||['upper','lower'].some(a=>dirty(a)||jobs[a].busy);}
function edited(a){jobs[a].revision++;updateTeeth();updateExport();$('decode-status').classList.remove('error');$('decode-status').textContent='参数已修改，稍后重算；当前仍显示上一版网格。';
 clearTimeout(jobs[a].timer);if(ready)jobs[a].timer=setTimeout(()=>submit(a),550);
}
async function api(path,options){const r=await fetch(path,options);const d=await r.json();if(!r.ok)throw Error(d.error||`HTTP ${r.status}`);return d;}
async function submit(a){
 clearTimeout(jobs[a].timer);if(!ready)return;
 const state=jobs[a],revision=state.revision,token=++state.token,snapshot=JSON.parse(JSON.stringify(q[a]));state.busy=true;updateExport();
 $('decode-status').classList.remove('error');$('decode-status').textContent=`${a==='upper'?'上颌':'下颌'}正在重新解码…`;
 try{const job=await api('/api/decode',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({arch:a,q:snapshot})});
  async function poll(){
   if(token!==state.token)return;
   try{const current=await api('/api/job?id='+job.id);
    if(current.status==='error')throw Error(current.error);
    if(current.status==='done'){
     if(revision===state.revision){const data=await api('/api/mesh?job='+job.id);if(token!==state.token||revision!==state.revision)return;installMesh(a,data);
      $('decode-status').textContent=`${a==='upper'?'上颌':'下颌'}已更新 · ${data.seconds.toFixed(1)} 秒 · ${data.count.triangles.toLocaleString()} 面`;
     }
     state.busy=false;updateExport();return;
    }
    if(current.status==='superseded'){state.busy=false;updateExport();return;}
    setTimeout(poll,700);
   }catch(e){if(token===state.token){state.busy=false;updateExport();fail(e.message+'；当前保留上次网格。');}}
  }poll();
 }catch(e){if(token===state.token){state.busy=false;updateExport();fail(e.message);}}
}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function worldPosition(v,p,lower){let x=v[0]+p.x,y=v[1]+p.y,z=v[2]+p.z;if(lower){const angle=opening*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle),dy=y-p.hinge_y;y=c*dy-s*z+p.hinge_y;z=s*dy+c*z;}return[x,y,z];}
function exportMesh(){if($('export-mesh').disabled)return;
 const lines=['# DMM R4; actual decoded geometry, millimeters. Display-only jaw placement.',`# Opening ${opening} degrees. Not calibrated occlusion.`];let offset=0;
 for(const arch of ['upper','lower']){const data=meshData[arch];lines.push('o '+arch);for(const v of data.vertices)lines.push('v '+worldPosition(v,data.placement,arch==='lower').map(x=>x.toFixed(5)).join(' '));for(const f of data.faces)lines.push('f '+f.map(i=>i+offset+1).join(' '));offset+=data.vertices.length;}
 download('dmm_dual_arch.obj',lines.join('\n')+'\n','text/plain');
}
function buildUI(){
 for(const a of ['upper','lower'])for(const k of orders[a]){const b=document.createElement('button');b.id=`tooth-${k}`;b.className='tooth';b.textContent=k;b.title=`${a==='upper'?'上颌':'下颌'} FDI ${k}`;b.onclick=()=>selectTooth(a,k);$(a+'-teeth').append(b);}
 for(let i=0;i<10;i++){
  const row=document.createElement('div');row.className='slider-row';const label=document.createElement('label');label.htmlFor=`q-${i}`;label.textContent=`q${i+1}`;
  const range=document.createElement('input');range.type='range';range.min=-3;range.max=3;range.step=.1;range.value=0;range.id=`q-${i}`;range.disabled=true;range.setAttribute('aria-label',`latent q${i+1}`);
  const number=document.createElement('input');number.type='number';number.min=-3;number.max=3;number.step=.1;number.value='0.0';number.id=`value-${i}`;number.disabled=true;number.setAttribute('aria-label',`latent q${i+1} 数值`);
  range.oninput=()=>{q[selectedArch][selectedTooth][i]=Number(range.value);number.value=Number(range.value).toFixed(1);edited(selectedArch);};
  number.oninput=()=>{if(number.value==='')return;const v=Number(number.value);if(!Number.isFinite(v))return;const clipped=Math.max(-3,Math.min(3,v));if(clipped!==v)number.value=clipped;range.value=clipped;q[selectedArch][selectedTooth][i]=clipped;edited(selectedArch);};number.onblur=()=>{number.value=Number(q[selectedArch][selectedTooth][i]).toFixed(1);};
  row.append(label,range,number);$('sliders').append(row);
 }
 selectTooth('upper',11);
 $('opening').oninput=e=>{opening=Number(e.target.value);$('opening-value').textContent=opening+'°';draw();};
 $('wire').onchange=e=>{wire=e.target.checked;draw();};$('color').onchange=e=>{color=e.target.checked;draw();};
 $('reset-view').onclick=()=>{setVisibility('both');setView('oblique');};
 document.querySelectorAll('#views button').forEach(b=>b.onclick=()=>setView(b.dataset.view));
 document.querySelectorAll('#visibility button').forEach(b=>b.onclick=()=>{setVisibility(b.dataset.mode);setView('oblique');});
 $('reset-tooth').onclick=()=>{q[selectedArch][selectedTooth]=Array(10).fill(0);selectTooth(selectedArch,selectedTooth);edited(selectedArch);};
 $('reset-all').onclick=()=>{for(const a of ['upper','lower']){for(const k of orders[a])q[a][k]=Array(10).fill(0);edited(a);}selectTooth(selectedArch,selectedTooth);};
 $('apply').onclick=()=>submit(selectedArch);
 $('export-params').onclick=()=>download('dmm_latent_parameters.json',JSON.stringify({schema:'dmm-latent-viewer-v1',models:metadata?.models,q,opening_degrees:opening,display_placement:metadata?.placement,display_matches_parameters:ready&&!dirty('upper')&&!dirty('lower')},null,2),'application/json');
 $('export-mesh').onclick=exportMesh;
 canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);lastPointer=[e.clientX,e.clientY];};
 canvas.onpointermove=e=>{if(!lastPointer)return;const dx=e.clientX-lastPointer[0],dy=e.clientY-lastPointer[1];if(e.shiftKey){pan[0]+=dx*2/canvas.clientWidth;pan[1]-=dy*2/canvas.clientHeight;}else{yaw-=dx*.008;elevation=Math.max(-Math.PI/2,Math.min(Math.PI/2,elevation+dy*.008));}lastPointer=[e.clientX,e.clientY];draw();};
 canvas.onpointerup=canvas.onpointercancel=()=>lastPointer=null;
 canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});canvas.ondblclick=()=>setView('oblique');
 new ResizeObserver(draw).observe(canvas);
}
async function connect(){
 try{const status=await api('/api/status');
  if(status.error)throw Error(status.error);
  $('load-progress').value=status.progress;$('loading-message').textContent=status.message;
  if(!status.ready){setTimeout(connect,1200);return;}
  metadata=status.metadata;
  for(const a of ['upper','lower'])installMesh(a,await api('/api/mesh?arch='+a));
  ready=true;$('loading').classList.add('hidden');$('connection').textContent='本机 CUDA · 已连接';$('connection').className='badge ready';
  $('apply').disabled=false;document.querySelectorAll('#sliders input').forEach(i=>i.disabled=false);
  $('model-info').textContent=`上颌 epoch ${metadata.models.upper.epoch} / 下颌 epoch ${metadata.models.lower.epoch}。当前 ${metadata.resolution}³ 提取；每个组件独立缓存，变化后重新合成整颌表面。`;
  $('decode-status').textContent='均值形状已载入，可以选择牙位调节。';updateExport();updateCount();
  setInterval(async()=>{try{const state=await api('/api/status');if(jobs.upper.busy||jobs.lower.busy){$('decode-status').textContent=state.message+' · '+Math.round(state.progress*100)+'%';} $('connection').textContent='本机 CUDA · 已连接';$('connection').className='badge ready';}catch(e){$('connection').textContent='模型服务已断开';$('connection').className='badge error';}},2000);
 }catch(e){$('loading-message').textContent=e.message+'。请检查本机服务日志后刷新页面。';$('connection').textContent='连接失败';$('connection').className='badge error';fail(e.message);}
}
try{initGL();buildUI();connect();}catch(e){$('loading-message').textContent=e.message;fail(e.message);}
