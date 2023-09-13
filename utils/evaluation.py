import torch
import os


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0) # beacuse the pixel value of obj is 1

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N

    return iou


def save_mask(pred_masks, save_base_path, video_name_list):
    # pred_mask: [bs*5, 1, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
    
    pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx] # [5, 1, 224, 224]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png"%(video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')