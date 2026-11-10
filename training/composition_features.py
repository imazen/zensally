"""Experimental compact geometry features; no aesthetic grades or face certification."""
import math
import numpy as np

NAMES=(
 'pose_count','pose_confidence','joints_retained','head_joints_retained','shoulders_retained','hips_retained','hands_retained','knees_retained','feet_retained','whole_pose_retained','upper_arm_endpoint_cut','forearm_endpoint_cut','thigh_endpoint_cut','shin_endpoint_cut','torso_endpoint_cut','joint_margin',
 'gaze_count','gaze_in_frame','gaze_head_retained','gaze_target_retained','gaze_pair_retained','gaze_ray_retained','headroom','looking_room',
 'scan_first_retained','scan_first_three_retained','scan_all_retained','scan_prefix_retained','scan_complete_paths','scan_transitions_retained','scan_centroid_x','scan_centroid_y')

def inside(points,box):
 x0,y0,x1,y1=box
 return (points[:,0]>=x0)&(points[:,0]<=x1)&(points[:,1]>=y0)&(points[:,1]<=y1)

def fraction(values,weights):
 total=float(weights.sum())
 return float(np.sum(values*weights)/total) if total>0 else 0.

def exit_t(origin,vector,box):
 if not inside(np.asarray([origin]),box)[0]:return 0.
 x,y=origin;dx,dy=vector;x0,y0,x1,y1=box;limits=[]
 if dx>1e-12:limits.append((x1-x)/dx)
 if dx< -1e-12:limits.append((x0-x)/dx)
 if dy>1e-12:limits.append((y1-y)/dy)
 if dy< -1e-12:limits.append((y0-y)/dy)
 return max(0.,min(limits)) if limits else 0.

def candidate_features(boxes,pose,gaze,scan):
 boxes=np.asarray(boxes,dtype=np.float64)
 if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all() or not ((boxes[:,:2]>=0).all() and (boxes[:,2:]<=1).all() and (boxes[:,2:]>boxes[:,:2]).all()):raise ValueError('finite normalized nonempty crop boxes required')
 poses=pose['landmarks'];people=gaze['people'];paths=scan['paths']
 if any(len(p)!=33 for p in poses) or len(poses)>8 or len(people)>16:raise ValueError('teacher geometry shape exceeds protocol')
 if len(paths)!=4 or any(len(p)!=6 for p in paths):raise ValueError('four independent six-fixation paths required')
 result=[]
 for box in boxes:
  x0,y0,x1,y1=box;values=[0.]*32;values[0]=len(poses)/8;all_xy=[];all_weight=[];keep_all=[];cut_lists=[[],[],[],[],[]]
  subset_values=[[] for _ in range(6)];subset_weights=[[] for _ in range(6)]
  for person in poses:
   points=np.asarray([[j['x'],j['y']] for j in person],float);confidence=np.asarray([(j['visibility'] or 0)*(j['presence'] or 0) for j in person],float)
   if not np.isfinite(points).all() or not np.isfinite(confidence).all() or not ((confidence>=0)&(confidence<=1)).all():raise ValueError('invalid pose confidence/point')
   weight=confidence*(confidence>=.25);kept=inside(points,box);all_xy.extend(points);all_weight.extend(weight)
   keep_all.append(float(kept[weight>0].all()) if (weight>0).any() else 0.)
   for i,indices in enumerate([range(11),[11,12],[23,24],range(15,23),[25,26],range(27,33)]):
    idx=list(indices);subset_values[i].extend(kept[idx]);subset_weights[i].extend(weight[idx])
   for i,segments in enumerate([[(11,13),(12,14)],[(13,15),(14,16)],[(23,25),(24,26)],[(25,27),(26,28)],[(11,23),(12,24)]]):
    for a,b in segments:
     if weight[a]>0 and weight[b]>0:cut_lists[i].append(float(kept[a]!=kept[b]))
  if all_weight:
   points=np.asarray(all_xy);weight=np.asarray(all_weight);values[1]=float(weight.mean());values[2]=fraction(inside(points,box),weight)
   for i in range(6):values[3+i]=fraction(np.asarray(subset_values[i]),np.asarray(subset_weights[i]))
   values[9]=float(np.mean(keep_all))
   for i,cuts in enumerate(cut_lists):values[10+i]=float(np.mean(cuts)) if cuts else 0.
   margin=np.minimum.reduce([(points[:,0]-x0)/(x1-x0),(x1-points[:,0])/(x1-x0),(points[:,1]-y0)/(y1-y0),(y1-points[:,1])/(y1-y0)])
   values[15]=fraction(np.clip(margin,-.5,.5)+.5,weight)
  values[16]=len(people)/16
  if people:
   centers=np.asarray([[(p['head'][0]+p['head'][2])/2,(p['head'][1]+p['head'][3])/2] for p in people]);targets=np.asarray([p['target'] for p in people]);w=np.asarray([p['in_frame_score'] for p in people]);hk=inside(centers,box);tk=inside(targets,box)
   if not np.isfinite(w).all() or not ((w>=0)&(w<=1)).all():raise ValueError('invalid in-frame score')
   values[17]=float(w.mean());values[18]=float(hk.mean());values[19]=fraction(tk,w);values[20]=fraction(hk&tk,w)
   values[21]=fraction(np.asarray([min(1,exit_t(c,t-c,box)) for c,t in zip(centers,targets)]),w)
   values[22]=float(np.mean([max(0,min(1,(p['head'][1]-y0)/(y1-y0))) if k else 0 for p,k in zip(people,hk)]))
   room=[]
   for c,t in zip(centers,targets):
    original=exit_t(c,t-c,[0,0,1,1]);room.append(min(1,exit_t(c,t-c,box)/original) if original>0 else 0)
   values[23]=fraction(np.asarray(room),w)
  xy=np.asarray([[[p['x'],p['y']] for p in path] for path in paths]);flat=xy.reshape(-1,2)
  if not np.isfinite(flat).all() or not ((flat>=0)&(flat<=1)).all():raise ValueError('invalid normalized scanpath coordinates')
  kept=inside(flat,box).reshape(4,6);values[24]=float(kept[:,0].mean());values[25]=float(kept[:,:3].mean());values[26]=float(kept.mean());values[27]=float(np.mean([np.argmax(~k)/6 if (~k).any() else 1 for k in kept]));values[28]=float(kept.all(-1).mean());values[29]=float((kept[:,:-1]&kept[:,1:]).mean())
  retained=flat[kept.flatten()];values[30:32]=((retained.mean(0)-[x0,y0])/[x1-x0,y1-y0]).tolist() if len(retained) else [.5,.5]
  if not np.isfinite(values).all() or min(values)<0 or max(values)>1+1e-9:raise ValueError('feature outside normalized bounds')
  result.append(values)
 return np.asarray(result,dtype=np.float32)
