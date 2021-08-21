from mmseg.datasets.pipelines.transforms import CLAHE
import os.path as osp
import tempfile

import mmcv

from mmcv.utils import print_log
from mmcv.utils import mkdir_or_exist
from mmseg.utils import get_root_logger
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

import cv2


@DATASETS.register_module()
class MirrorDataset(CustomDataset):
    """ICIP mirror dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    # CLASSES = ('mirror', 'trans')
    CLASSES = ('background', 'trans', 'mirror')

    PALETTE = [[6, 230, 230], [120, 120, 120], [180, 120, 120]]
    DEPTH_COLOR_MAP = cv2.COLORMAP_JET

    def __init__(self, depth_dir=None, depth_suffix='.png', **kwargs):
        super(MirrorDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

        self.depth_dir = depth_dir
        if self.data_root is not None:
            if not osp.isabs(self.depth_dir):
                self.depth_dir = osp.join(self.data_root, self.depth_dir)
        self.depth_suffix = depth_suffix
        self.update_depth_to_img_infos(self.depth_dir, self.depth_suffix)

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            # The  index range of official requirement is from 0 to 150.
            # But the index range of output is from 0 to 149.
            # That is because we set reduce_zero_label=True.
            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for ade20k evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: '
        #     f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None

        result_files = self.results2img(results, imgfile_prefix, to_label_id)
        return result_files, tmp_dir

    def update_depth_to_img_infos(self, depth_dir, depth_suffix):
        if depth_dir is None:
            return
        for img_info in self.img_infos:
            img_name = img_info['filename'].split('.')[0]
            depth_map = img_name + depth_suffix
            img_info['depth'] = dict(depth_map=depth_map)

    def get_depth_info(self, idx):
        if 'depth' in self.img_infos[idx]:
            return self.img_infos[idx]['depth']
        return None

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        depth_info = self.get_depth_info(idx)
        if depth_info:
            results['depth_info'] = depth_info
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        depth_info = self.get_depth_info(idx)
        if depth_info:
            results['depth_info'] = depth_info
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map
        if self.depth_dir:
            results['depth_fields'] = []
            results['depth_prefix'] = self.depth_dir

    def vis_results(self, runner, results, vis_cfg=None):
        current = runner.iter
        save_path = osp.join(runner.work_dir, 'sample', f'iter_{current}')
        mkdir_or_exist(save_path)

        vis_depth = vis_cfg.get('vis_depth', False) if vis_cfg else False
        max_num = vis_cfg.get('max_num', -1) if vis_cfg else -1
        vis_idx = [idx for idx in range(len(self.img_infos))]
        if max_num != -1:
            import random
            random.shuffle(vis_idx)
            vis_idx = vis_idx[:max_num]

        if vis_depth:
            assert hasattr(
                self, 'depth_dir'
            ), '`depth_dir` is must for depth image visualization.'

        print_log(f'Save {max_num} results, please wait...')
        for idx in vis_idx:
            img_info = self.img_infos[idx]
            pred_seg_map = results[idx]

            gt_seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            img = osp.join(self.img_dir, img_info['filename'])

            img_np = np.array(Image.open(img))
            gt_seg_np = np.array(Image.open(gt_seg_map))

            color_gt_seg = np.zeros(
                (gt_seg_np.shape[0], gt_seg_np.shape[1], 3))
            color_pred_seg = np.zeros(
                (pred_seg_map.shape[0], pred_seg_map.shape[1], 3))

            for label, color in enumerate(self.PALETTE):
                color_gt_seg[gt_seg_np == label, :] = color
                color_pred_seg[pred_seg_map == label, :] = color

            color_pred_seg = mmcv.imresize(
                color_pred_seg, (color_gt_seg.shape[1], color_gt_seg.shape[0]))

            if vis_depth:
                # IMG | DEPTH
                # GT  | PRED
                depth_map = osp.join(self.depth_dir,
                                     img_info['depth']['depth_map'])

                # depth_np = np.array(Image.open(depth_map))[..., None]
                # depth_np = np.concatenate([depth_np, depth_np, depth_np],
                #                           axis=2).astype(np.uint8)
                depth_np = cv2.imread(depth_map)
                color_depth = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_np, alpha=15),
                    self.DEPTH_COLOR_MAP)
                inp_vis_res = np.concatenate([img_np, color_depth], axis=1)
                out_vis_res = np.concatenate([color_gt_seg, color_pred_seg],
                                             axis=1)
                vis_res = np.concatenate([inp_vis_res, out_vis_res],
                                         axis=0).astype(np.uint8)
            else:
                # IMG | GT | PRED
                vis_res = np.concatenate(
                    [img_np, color_gt_seg, color_pred_seg],
                    axis=1).astype(np.uint8)
            img_name = img_info['filename'].split('.')[0]
            Image.fromarray(vis_res).save(
                osp.join(save_path, f'{img_name}.png'))
            # vis_list.append(dict(img=vis_res, name=img_info['filename']))

        print_log('Save finished.')
