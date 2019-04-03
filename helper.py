import fastai
from fastai.vision import *

def get_segmentation_model(model, classes=[]):
    src = (SegmentationItemList.from_folder(path='data', convert_mode='L')
       .split_none()
       .label_from_func(lambda x: Path('data/label/0.png'), classes=classes)
      )

    data = src.databunch(no_check=True).normalize(imagenet_stats)
    #learn = unet_learner(data, models.resnet34).to_fp16()
    learn = unet_learner(data, models.resnet34)
    learn.load(model)
    return learn

def pred_details(pred, stamps):
    pred_n = pred[1].numpy().squeeze()
    pred_n = pred_n.sum(axis=0)
    mask = np.where(pred_n < 50, 0, 1)
    idx = np.where(mask!=0)[0]
    mask_group = np.split(mask[idx],np.where(np.diff(idx)!=1)[0]+1)
    stamp_group = np.split(stamps[idx],np.where(np.diff(idx)!=1)[0]+1)

    final_group = []
    for k, v in enumerate(mask_group):
        if sum(v) > 14:
            x = [k,int(str(sum(v))), stamp_group[k][0], stamp_group[k][-1]]
            final_group.append(x)

    return final_group

def build_input(vals, norm):

    vals = np.array(vals)
    vals = np.round( (vals / norm) * 100 )
    vals = vals.astype(int)
    data = np.full([3600,100],255).astype(dtype=np.uint8)

    for k, v in enumerate(vals):

        if k == len(vals)-1:
            n = vals[0]
        else:
            n = vals[k+1]

        if v > n:
            topv = 100 - v
            botv = n
            midv = v - n
        elif n > v:
            topv = 100 - n
            botv = v
            midv = n - v
        else:
            if v == 100:
                midv = 1
                topv = 0
                botv = 99
            else:
                midv = 1
                topv = 99 - v
                botv = v

        start = np.full(topv, 255)
        middle = np.full(midv, 0)
        bottom = np.full(botv, 255)

        total = np.concatenate((start,middle,bottom))
        total = total.astype(int)
        data[k] = total

    return torch.tensor(data.transpose()[None], dtype=torch.float)/255
