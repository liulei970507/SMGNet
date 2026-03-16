from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import random
import numpy as np
import math
class QATCLSDB50TrackActor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
    def softmax(self, x):
        """
        Compute the softmax function for each row of the input x.
        Arguments:
        x -- A N dimensional vector or M x N dimensional numpy matrix.
        Return:
        x -- You are allowed to modify x in-place
        """
        orig_shape = x.shape
    
        if len(x.shape) > 1:
            # Matrix
            # exp_minmax = lambda x: np.exp(x - np.max(x))
            exp_minmax = lambda x: np.exp(50.0*(x - np.max(x)))
            # torch.exp(
            denom = lambda x: 1.0 / np.sum(x)
            x = np.apply_along_axis(exp_minmax,1,x)
            denominator = np.apply_along_axis(denom,1,x) 
            
            if len(denominator.shape) == 1:
                denominator = denominator.reshape((denominator.shape[0],1))
            
            x = x * denominator
        else:
            # Vector
            x_max = np.max(x)
            x = x - x_max
            # numerator = np.exp(x)
            numerator = np.exp(50.0*x)
            denominator =  1.0 / np.sum(numerator)
            x = numerator.dot(denominator)
            
        assert x.shape == orig_shape
        return x
    
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['visible']['template_images']) == 1
        assert len(data['visible']['search_images']) == 1

        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])  # (batch, 3, 128, 128)        
        
        search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['visible']['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        out_dict = self.net(template=[template_img_v, template_img_i],
                            search=[search_img_v, search_img_i],
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        pred_boxes_rgb = pred_dict['pred_boxes_rgb']
        pred_boxes_tir = pred_dict['pred_boxes_tir']
        pred_weight = pred_dict['w']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_rgb = box_cxcywh_to_xyxy(pred_boxes_rgb).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_tir = box_cxcywh_to_xyxy(pred_boxes_tir).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_rgb, iou_rgb = self.objective['giou'](pred_boxes_vec_rgb, gt_boxes_vec)  # (BN,4) (BN,4)
            giou_loss_tir, iou_tir = self.objective['giou'](pred_boxes_vec_tir, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_rgb, iou_rgb = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            giou_loss_tir, iou_tir = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss_rgb = self.objective['l1'](pred_boxes_vec_rgb, gt_boxes_vec)  # (BN,4) (BN,4)
        l1_loss_tir = self.objective['l1'](pred_boxes_vec_tir, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
            location_loss_rgb = self.objective['focal'](pred_dict['score_map_rgb'], gt_gaussian_maps)
            location_loss_tir = self.objective['focal'](pred_dict['score_map_tir'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
            location_loss_rgb = torch.tensor(0.0, device=l1_loss.device)
            location_loss_tir = torch.tensor(0.0, device=l1_loss.device)
        # compute weight prediction loss
        # gt weight
        gt_mr_sec = []
        for i in range(iou_rgb.shape[0]):
            # gt_mr_sec.append(torch.nn.functional.softmax(torch.Tensor([iou_rgb[i].detach(),iou_tir[i].detach()]),dim=0)[0].item())
            # import pdb
            # pdb.set_trace()
            gt_mr_sec.append(self.softmax(np.array([iou_rgb[i].detach().cpu(),iou_tir[i].detach().cpu()]))[0])
        
        
        rounded_numbers = [round(num, 4) for num in gt_mr_sec] 
        gt_mr_s = []
        for thr in rounded_numbers:
            index = sum(i/16<thr for i in range(16))
            label = [1 if i<index else 0 for i in range(16)]
            gt_mr_s.append(label)
        # compute weight prediction loss
        gt_mr = torch.tensor(gt_mr_s,dtype=pred_weight[0].dtype).to(pred_weight[0].device)
        if random.random()>0.995:
            print('真值', rounded_numbers)
            print('预测', pred_weight[0].squeeze(-1).sigmoid().mean(dim=1).detach())
        p_tag_loss = 0.
        for predicted_values in pred_weight: 
            p_tag_loss += self.objective['p_tag'](predicted_values.squeeze(-1), gt_mr)
            
        # weighted sum
        loss = self.loss_weight['giou'] * (giou_loss + giou_loss_rgb + giou_loss_tir) + self.loss_weight['l1'] * (l1_loss + l1_loss_rgb + l1_loss_tir) + self.loss_weight['focal'] * (location_loss + location_loss_rgb + location_loss_tir) + self.loss_weight['p_tag'] * p_tag_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            mean_iou_rgb = iou_rgb.detach().mean()
            mean_iou_tir = iou_tir.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/giou_rgb": giou_loss_rgb.item(),
                      "Loss/giou_tir": giou_loss_tir.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/l1_rgb": l1_loss_rgb.item(),
                      "Loss/l1_tir": l1_loss_tir.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/location_rgb": location_loss_rgb.item(),
                      "Loss/location_tir": location_loss_tir.item(),
                      "IoU": mean_iou.item(),
                      "IoU_rgb": mean_iou_rgb.item(),
                      "IoU_tir": mean_iou_tir.item(),
                      'p_tag': p_tag_loss.item()}
            return loss, status
        else:
            return loss
