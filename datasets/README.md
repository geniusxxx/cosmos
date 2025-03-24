# Data Preparation for Downstream Tasks
### Datasets list:
- [MSCOCO](#coco)
- [Flickr30k](#flickr)
- [ImageNet](#imagenet)
- [Other Classification datasets](#others)
- [Segmentation datasets](#segmentation)


## Image-Text Retrieval Task

### <span id ='coco'> MSCOCO dataset
```
$coco/
|–– images/
|–––– val2017/
|–––––– 000000134722.jpg
|–––––– 000000177015.jpg
|–––––– ...
|–– annotations/
|–––– captions_val2017.json
```
Step 1. Download validation images from [COCO 2017 Val Images](https://cocodataset.org/#download), unzip them to `coco/images/val2017`.

Step 2. Download the 2017 Val annotations, place it under `coco/annotations/captions_val2017.json`.

### <span id ='flickr'> Flickr30K dataset
```
$flickr30k-images/
|––  2217728745.jpg 
|––  2217728745.jpg
|––  ...
|––  flickr30k_val.json
|––  flickr30k_test.json
```
Step 1. Download  [flickr30k dataset](https://huggingface.co/datasets/nlphuji/flickr30k), unzip them under `flickr30k-images/`, all the images and annotations files will be structured as above.

## Image Classification Task

### <span id ='imagenet'> ImageNet dataset
```
$imagenet/
|–– data/
|–––– val_images/
|–––––– n01440764/
|–––––––– ILSVRC2012_val_00000293_n01440764.JPEG
|–––––––– ILSVRC2012_val_00017699_n01440764.JPEG
|–––––––– ...
|–––––– n01871265/
|–––––––– ILSVRC2012_val_00000067_n01871265.JPEG
|–––––––– ILSVRC2012_val_00017361_n01871265.JPEG 
|–––––––– ...
```

Step 1. Download validation data `val_images.tar.gz` from [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k), and unzip them to `imagenet/data/val_images`.
You can manually download the `imagenet-1k/data/val_images.tar.gz` or use this command. `huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir /directory/to/your/dataset/`.

Step 2. Change [source_dir](imagenet_organize.py#L5) in `imagenet_organize.py` according to your val_images folder. Then, run `imagenet_organize.py` to organize the image in the above format. 

### <span id ='others'> Other Classification datasets

Other classification datasets include `["food101", "cifar10", "cifar100", "sun397", "stanford_car", "aircraft", "dtd", "pets", "caltech101", "flowers"]`.

Please set appropriate [dataset_root](/src/dataloaders/utils.py#L17) in `src/dataloaders/utils.py` to save classification datasets. 

Then, `torchvision.datasets` will automatically download the datatsets in `dataset_root` during inference.  


## Semantic Segmentation Task

### <span id ='segmentation'> Segmentation datasets

We followed the evaluation scheme and config files provided by [SCLIP](https://github.com/wangf3014/SCLIP) as shown [here](/src/training/seg_configs).

Our segmentation configs include benchmarks with background `['cfg_voc21.py', 'cfg_context60.py', 'cfg_coco_object.py']` and without background `['cfg_voc20.py', 'cfg_city_scapes.py', 'cfg_context59.py', 'cfg_ade20k.py', 'cfg_coco_stuff164k.py']`.

Please follow the dataset preparation instruction provided by [SCLIP](https://github.com/wangf3014/SCLIP) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download the following datasets: `["VOCdevkit/VOC2012", "VOCdevkit/VOC2010", "coco_stuff164k", "cityscapes, "ade"]`.

Then, change the `data_root` in each segmentation config according to the dataset location. For example, this is [root_dir](/src/training/seg_configs/cfg_ade20k.py#L12) for `cfg_ade20k.py`.
