import math

from lib.models.qat_cls_weight_guide_trans_encoder_track import build_qat_cls_weight_guide_trans_encoder_track
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from collections import OrderedDict
import numpy as np

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):

    if overlap_range is None and scale_range is None:
        return generator(bbox, n)

    else:
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 10240000:
            samples_ = generator(bbox, remain*factor)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:,2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx,:]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor*2

        return samples
    
class SampleGenerator():
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        self.type = type
        self.img_size = np.array(img_size) # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid

    def __call__(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')

        # (center_x, center_y, w, h)
        sample = np.array([bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None,:],(n,1))

        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n,1)*2-1
            samples[:,2:] *= self.aspect_f ** np.concatenate([ratio, -ratio],axis=1)

        # sample generation
        if self.type=='gaussian':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
            samples[:,2:] *= self.scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)

        elif self.type=='uniform':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        elif self.type=='whole':
            m = int(2*np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
            xy = np.random.permutation(xy)[:n]
            samples[:,:2] = bb[2:]/2 + xy * (self.img_size-bb[2:]/2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        # adjust bbox range
        samples[:,2:] = np.clip(samples[:,2:], 5, self.img_size-5.)
        if self.valid:
            samples[:,:2] = np.clip(samples[:,:2], samples[:,2:]/2, self.img_size-samples[:,2:]/2-1)
        else:
            samples[:,:2] = np.clip(samples[:,:2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:,:2] -= samples[:,2:]/2

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f

    def get_trans_f(self):
        return self.trans_f
    
class TBSITrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TBSITrack, self).__init__(params)
        network = build_qat_cls_weight_guide_trans_encoder_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.gt = np.loadtxt('/data1/Datasets/Tracking/RGBT234/car37/init.txt',delimiter=',')
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        
        if self.frame_id==583:
            for cyc in range(2):
                # cyc=1
                pos_examples = gen_samples(SampleGenerator('gaussian', (H, W), 0.1, 1.2), self.gt[self.frame_id], 128, [0.4,1])
                pos_feature_rgb = None
                pos_feature_t = None
                pos_feature_rgb_ori = None
                pos_feature_t_ori = None
                
                pos_idx = 0
                for pos_index in pos_examples:
                    pos_idx = pos_idx+1
                    print('pos', pos_idx, pos_index)
                    # pos 1 [412.65848  200.92911   29.054426  69.94585 ]
                    # import pdb
                    # pdb.set_trace()
                    self.state = pos_index.tolist()
                    # torch.Tensor([pos_index[1] + (pos_index[3] - 1)/2, pos_index[0] + (pos_index[2] - 1)/2])
                    # self.target_sz = torch.Tensor([pos_index[3], pos_index[2]])
                    
                    x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
                    search = self.preprocessor.process(x_patch_arr, x_amask_arr)

                    with torch.no_grad():
                        x_dict = search
                        # merge the template and the search
                        # run the transformer
                        out_dict = self.network.forward(
                            template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]], search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]], ce_template_mask=self.box_mask_z)
            
                    backbone_feat = out_dict['backbone_feat_ori']
                    backbone_feat_enhanced = out_dict['backbone_feat']
                    # import pdb
                    # pdb.set_trace()
                    backbone_feat_v_ori = backbone_feat[:, :320, :]
                    backbone_feat_v_enhanced = backbone_feat_enhanced[:, :320, :]
                    backbone_feat_i_ori = backbone_feat[:, 320:, :]
                    backbone_feat_i_enhanced = backbone_feat_enhanced[:, 320:, :]
                    if pos_idx==1:
                        pos_feature_rgb = backbone_feat_v_enhanced.data.cpu().numpy().reshape(1,-1)
                        pos_feature_rgb_ori = backbone_feat_v_ori.data.cpu().numpy().reshape(1,-1)
                        pos_feature_t = backbone_feat_i_enhanced.data.cpu().numpy().reshape(1,-1)
                        pos_feature_t_ori = backbone_feat_i_ori.data.cpu().numpy().reshape(1,-1)
                    else:
                        pos_feature_rgb = np.vstack((pos_feature_rgb,backbone_feat_v_enhanced.data.cpu().numpy().reshape(1,-1)))
                        pos_feature_rgb_ori = np.vstack((pos_feature_rgb_ori,backbone_feat_v_ori.data.cpu().numpy().reshape(1,-1)))
                        pos_feature_t = np.vstack((pos_feature_t,backbone_feat_i_enhanced.data.cpu().numpy().reshape(1,-1)))
                        pos_feature_t_ori = np.vstack((pos_feature_t_ori,backbone_feat_i_ori.data.cpu().numpy().reshape(1,-1)))
                
                # 负样本
                neg_examples = gen_samples(SampleGenerator('gaussian', (H, W), 0.5, 2, 2), self.gt[self.frame_id], 128, [0.0,0.6])
                neg_feature_rgb = None
                neg_feature_t = None
                neg_feature_rgb_ori = None
                neg_feature_t_ori = None
                neg_idx = 0
                for neg_index in neg_examples:
                    neg_idx = neg_idx+1
                    print('neg', neg_idx, neg_index)
                    # self.state = torch.Tensor([neg_idx[1] + (neg_idx[3] - 1)/2, neg_idx[0] + (neg_idx[2] - 1)/2])
                    # self.target_sz = torch.Tensor([neg_idx[3], neg_idx[2]])
                    self.state = neg_index.tolist()
                    x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
                    search = self.preprocessor.process(x_patch_arr, x_amask_arr)


                    with torch.no_grad():
                        x_dict = search
                        # merge the template and the search
                        # run the transformer
                        out_dict = self.network.forward(
                            template=[self.z_dict1.tensors[:,:3,:,:],self.z_dict1.tensors[:,3:,:,:]], search=[x_dict.tensors[:,:3,:,:], x_dict.tensors[:,3:,:,:]], ce_template_mask=self.box_mask_z)
            
                    backbone_feat = out_dict['backbone_feat_ori']
                    backbone_feat_enhanced = out_dict['backbone_feat']
                    backbone_feat_v_ori = backbone_feat[:, :320, :]
                    backbone_feat_v_enhanced = backbone_feat_enhanced[:, :320, :]
                    backbone_feat_i_ori = backbone_feat[:, 320:, :]
                    backbone_feat_i_enhanced = backbone_feat_enhanced[:, 320:, :]
                    
                    if neg_idx==1:
                        neg_feature_rgb = backbone_feat_v_enhanced.data.cpu().numpy().reshape(1,-1)
                        neg_feature_rgb_ori = backbone_feat_v_ori.data.cpu().numpy().reshape(1,-1)
                        neg_feature_t = backbone_feat_i_enhanced.data.cpu().numpy().reshape(1,-1)
                        neg_feature_t_ori = backbone_feat_i_ori.data.cpu().numpy().reshape(1,-1)
                    else:
                        neg_feature_rgb = np.vstack((neg_feature_rgb,backbone_feat_v_enhanced.data.cpu().numpy().reshape(1,-1)))
                        neg_feature_rgb_ori = np.vstack((neg_feature_rgb_ori,backbone_feat_v_ori.data.cpu().numpy().reshape(1,-1)))
                        neg_feature_t = np.vstack((neg_feature_t,backbone_feat_i_enhanced.data.cpu().numpy().reshape(1,-1)))
                        neg_feature_t_ori = np.vstack((neg_feature_t_ori,backbone_feat_i_ori.data.cpu().numpy().reshape(1,-1)))
                # import pdb
                # pdb.set_trace()
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_pos_rgb_woweight'+str(cyc)+'.txt', pos_feature_rgb, delimiter=',')
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_neg_rgb_woweight'+str(cyc)+'.txt', neg_feature_rgb, delimiter=',')
                
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_pos_t_woweight'+str(cyc)+'.txt', pos_feature_t, delimiter=',')
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_neg_t_woweight'+str(cyc)+'.txt', neg_feature_t, delimiter=',')
                
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_pos_rgb_ori'+str(cyc)+'.txt', pos_feature_rgb_ori, delimiter=',')
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_neg_rgb_ori'+str(cyc)+'.txt', neg_feature_rgb_ori, delimiter=',')
                
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_pos_t_ori'+str(cyc)+'.txt', pos_feature_t_ori, delimiter=',')
                np.savetxt('/data1/Code/liulei/ACMMM2023Extension/TBSI/tsne_file/car37_583_neg_t_ori'+str(cyc)+'.txt', neg_feature_t_ori, delimiter=',')
                
        else:
            pass
        
        return {"target_bbox": [1,1,1,1],
                    'w':1}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TBSITrack
