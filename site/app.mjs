const $=id=>document.getElementById(id);
const el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n;};
const pct=value=>(value*100).toFixed(1)+"%";
const count=value=>new Intl.NumberFormat("en").format(value);
let map;
let requestId=0;
async function start(){
 const current=++requestId;
 const response=await fetch($("event-select").value);
 if(!response.ok)throw Error("Results unavailable");
 const data=await response.json();
 if(current!==requestId)return;
 const final=Boolean(data.checkpoint_sha256);
 $("event").textContent=(data.title||"France floods · January 2018")+" · "+data.event;
 $("coverage").textContent=data.test+" sampled tiles · 128 × 128 pixels per tile. "+
  (final?"The original checkpoint and both cutoffs were frozen before scoring this event.":"The event was held out from fitting, and an early result was inspected during development.")+
  (data.metrics.model.iou>data.metrics.threshold.iou?" The CNN scored higher on IoU.":" The radar threshold matched or exceeded the CNN on IoU.");
 if(map){map.remove();map=null;}
 $("comparison").replaceChildren(...Object.entries(data.metrics).map(([key,m])=>{
  const row=el("tr","");
  row.append(...[key==="model"?"Convolutional model":"Radar threshold",pct(m.iou),pct(m.precision),pct(m.recall),count(m.fp),count(m.fn)].map(x=>el("td",x)));
  return row;
 }));
 const rectangles=new Map();
 if(window.L){
  map=L.map("map",{scrollWheelZoom:false});
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{attribution:"© OpenStreetMap contributors",maxZoom:18}).addTo(map);
  for(const tile of data.tiles){
   const [w,s,e,n]=tile.bounds;
   const box=L.rectangle([[s,w],[n,e]],{color:"#26839c",weight:2}).addTo(map);
   box.on("click",()=>{$("tile").value=tile.id;render();});
   rectangles.set(tile.id,box);
  }
  map.fitBounds(L.featureGroup([...rectangles.values()]).getBounds(),{padding:[15,15],maxZoom:13});
 }else $("map").textContent="Map tiles are unavailable. Use the satellite tile picker.";
 function render(){
  const tile=data.tiles.find(item=>item.id===$("tile").value);
  $("tile-score").textContent="Model IoU "+pct(tile.model.iou)+" · "+count(tile.model.fp)+" false alarms · "+count(tile.model.fn)+" missed flood pixels";
  for(const [id,box] of rectangles)box.setStyle({color:id===tile.id?"#fa9554":"#26839c",weight:id===tile.id?3:2});
  $("images").replaceChildren(...[["radar","Sentinel-1 radar"],["model","Model prediction"],["reference","Reference mask"],["threshold","Radar threshold"],["errors","Where the model missed"]].map(([key,title])=>{
   const figure=el("figure",""),img=el("img","");
   img.src="tiles/"+tile.id+"-"+key+".png";img.alt=title+" for "+tile.id;img.width=128;img.height=128;
   figure.append(el("figcaption",title),img);
   return figure;
  }));
 }
 function options(){
  const sort=$("sort").value;
  const previous=$("tile").value;
  const tiles=[...data.tiles];
  if(sort!=="name")tiles.sort((a,b)=>sort==="missed"?b.model.fn-a.model.fn:b.model.fp-a.model.fp);
  $("tile").replaceChildren(...tiles.map(t=>new Option(t.id,t.id)));
  if(sort==="name"&&previous)$("tile").value=previous;
  render();
 }
 $("tile").value="";$("tile").onchange=render;$("sort").onchange=options;options();$("status").textContent="";
}
function loadEvent(){start().catch(()=>{$("status").textContent="The selected replay could not load. Try refreshing.";});}
$("event-select").onchange=loadEvent;
loadEvent();
