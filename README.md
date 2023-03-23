# SDM-RAN
![Examples](assets/examples.png)
This repository contains the demos for paper **Identify an Object Unseen Before At Once without Fine-tuning**.
## Abstract
Given one or a couple of photos of an object unseen before, humans can find it immediately in different scenes. Though the human brain mechanism behind this phenomenon is still not fully discovered, this work introduces a novel engineering realization of this task. It consists of two steps: (1) Generating a **Similarity Density Map (SDM)** by convoluting the scene image using the given object image patch(es) so that the highlight areas in SDM indicate the possible locations of the object; (2) Obtaining the object occupied areas in the scene through a **Region Align Network (RAN)**. RAN is constructed on a backbone of Deep Siamese Network (DSN) and different from the traditional DSNs, RAN aims to obtain the objects region according to the location and box differences between the ground truth and the predicted one around the highlight areas in SDM. With the pre-learning on the annotated labels given by traditional datasets of RAN, the proposed SDM-RAN can identify an object unseen before at once without fine-tuning or re-training. Experiments are carried out on COCO dataset for object detection, and FSC-147 dataset for object counting. The results indicate the proposed method outperforms the state-of-the-art methods on related tasks.
## Installation
+ Install PyTorch:
```
conda install pytorch=1.12.0 torchvision torchaudio cudatoolkit=11.7 -c pytorch
```
+ Install necessary packages with `requirements.txt`
```
pip install -r requirements.txt
```
The code was developed and tested with Python 3.8, Pytorch 1.12.0, and opencv 4.6.0
## Code directory structure
```
├── SDM-RAN
│   ├── FSOD-AO
│   │   ├── [scene_id]									
│   │   │   ├── [scene_id].aggregation.json
│   │   │   ├── [scene_id]_vh_clean_2.0.010000.segs.json
│   │   │   ├── [scene_id]_vh_clean_2.labels.ply
│   │   │   ├── [scene_id]_vh_clean_2.ply
│   ├── VCount
│   │   ├── [scene_id]								
│   │   │   ├── [scene_id]_vh_clean_2.ply
│   ├── scannetv2-labels.combined.tsv
```
