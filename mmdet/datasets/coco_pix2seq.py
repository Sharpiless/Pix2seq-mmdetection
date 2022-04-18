from mmdet.datasets.pipelines.compose import Compose
from .coco import CocoDataset, COCO
from .builder import DATASETS
from copy import deepcopy


@DATASETS.register_module()
class Pix2seqCocoDataset(CocoDataset):

    CLASSES = ('N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
               'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
               'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
               'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
               'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
               'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
               'toothbrush')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        super(CocoDataset, self).__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt
        )
        self.pipeline2 = Compose(pipeline)
        # debug
        # self.data_infos = self.data_infos[:100]
        # self.img_ids = self.img_ids[:100]

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = list(range(len(self.CLASSES)))

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data1, data2 = self.prepare_train_img(idx)
            if data1 is None or data2 is None:
                idx = self._rand_another(idx)
                continue
            elif not data1['gt_bboxes'].data.shape[0] or not data2['gt_bboxes'].data.shape[0]:
                idx = self._rand_another(idx)
                continue
            return data1, data2

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results1 = dict(img_info=img_info, ann_info=ann_info)
        results2 = deepcopy(results1)
        if self.proposals is not None:
            results1['proposals'] = self.proposals[idx]
            results2['proposals'] = self.proposals[idx]
        self.pre_pipeline(results1)
        self.pre_pipeline(results2)
        return self.pipeline(results1), self.pipeline2(results2)

