import torch
import math

def angulation(angten, device):
    sze = angten.size(dim=1)
    thetanew_prob = torch.zeros(sze).to(device)
    thetanew = torch.zeros(sze).to('cpu')
        
    for i in range(sze):
        thetanew_prob[i] = angten[i, :].max() #we have learned that there are 180 angular dimensions for each box(895), calculating their mean will achieve an average angle of each box in radians
        angle = (angten == thetanew_prob[i]).nonzero(as_tuple=False).to('cpu')
        angle = angle.squeeze()
        angle = angle.item()
        angle = (angle - 90) * (math.pi / 180) #according to the original csl definition, yv5obb uses -pi/2, +pi/2 angular definition method
        thetanew[i] = angle
    thetanew = thetanew.to(device)
    return thetanew
  
    
def KfiouPrep(box1, box2, device): #KFIOU tensor preparation
    if x1y1x2y2:
        xywhbox1 = xyxy2xywh(box1)
        xywhbox2 = xyxy2xywh(box2)
    else:
        xywhbox1 = box1
        xywhbox2 = box2
    newbox1 = torch.zeros_like(box1).to(device)
    sze = newbox1.size(dim=1)
    catcon1 = torch.zeros(1, sze).to(device)
    newbox1 = torch.cat((newbox1, catcon1), 0).to(device)
    newbox1[:4, :] = xywhbox1[:4, :]

    newbox1[4, :] = angulation(ptheta)

    #newbox1a = torch.zeros(sze, 5).to(device)
    newbox1a = torch.transpose(newbox1, 0, 1).to(device)
    #for i in range(sze):
        #for j in range(5):
            #newbox1a[i, j] = newbox1[j, i]

    newbox2 = torch.zeros_like(box2).to(device)
    sze = newbox2.size(dim=1)
    catcon2 = torch.zeros(1, sze).to(device)
    newbox2 = torch.cat((newbox2, catcon2), 0).to(device)
    newbox2[:4, :] = xywhbox2[:4, :]

    newbox2[4, :] = angulation(ttheta)

    #newbox2a = torch.zeros(sze, 5).to(device)
    newbox2a = torch.transpose(newbox2, 0, 1).to(device)
    #for i in range(sze):
        #for j in range(5):
            #newbox2a[i, j] = newbox2[j, i] 

    #the indexing method in KFIOU is vastly differnt from other IOUs. Thus, a transpose is required

    kfbox1, sigma1 = kfiou.xy_wh_r_2_xy_sigma(newbox1a)
    kfbox2, sigma2 = kfiou.xy_wh_r_2_xy_sigma(newbox2a)
