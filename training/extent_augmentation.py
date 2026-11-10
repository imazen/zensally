"""Full-content augmentations for weak extent fitting; source bounds stay explicit."""
import torch


def transform(images,targets,content,flip,brightness,contrast):
    """Reflect within each content rectangle, preserving the original padding."""
    batch,channels,height,width = images.shape
    bounds = (content*content.new_tensor([width,height,width,height])).round().long()
    x = torch.arange(width,device=images.device)[None]
    y = torch.arange(height,device=images.device)[None]
    inside_x = (x>=bounds[:,0:1]) & (x<bounds[:,2:3])
    inside_y = (y>=bounds[:,1:2]) & (y<bounds[:,3:4])
    reflected = bounds[:,0:1]+bounds[:,2:3]-1-x
    columns = torch.where(flip[:,None] & inside_x,reflected,x.expand(batch,-1))
    pixels = images.gather(3,columns[:,None,None,:].expand(batch,channels,height,width))
    pixels = (((pixels-.5)*contrast+.5)*brightness).clamp(0,1)
    pixels = pixels*(inside_y[:,:,None] & inside_x[:,None,:])[:,None]
    left,top,right,bottom = targets.unbind(-1)
    reflected_target = torch.stack((1-right,top,1-left,bottom),-1)
    return pixels,torch.where(flip[:,None],reflected_target,targets)


def augment(images,targets,content):
    batch = len(images)
    flip = torch.rand(batch,device=images.device)<.5
    brightness = .8+.4*torch.rand(batch,1,1,1,device=images.device)
    contrast = .8+.4*torch.rand(batch,1,1,1,device=images.device)
    return transform(images,targets,content,flip,brightness,contrast)
