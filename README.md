# DAAF re-ID

Codes for [Deep Attention Aware Feature Learning for Person Re-Identification](http://arxiv.org/abs/2003.00517)

We have tested the performance of our method on two typical networks: [TriNet](https://github.com/VisualComputingInstitute/triplet-reid) and [Bag of Tricks](https://github.com/michuanhaohao/reid-strong-baseline). The results are as follows (mAP (rank-1)).


| Method        | Market-1501      |  CUHK-03(labeled) | CUHK-03(Detected) | Duke-MTMC        |
| ------------- | ---------------- | ----------------- | ----------------- | ---------------- |
| [TriNet](https://github.com/VisualComputingInstitute/triplet-reid)       | 65.48 ( 82.51 )  | 46.39 ( 51.07 )  | 46.74 ( 51.93 ) | 53.50 ( 72.44 ) |
| [DAAF-TriNet](https://github.com/CYFFF/DAAF_re-id/tree/master/DAAF-Trinet)   | 70.40 ( 85.87 )  | 53.21 ( 58.23 )  | 51.98 ( 57.03 )  | 58.91 ( 76.63 )  |
| TriNet*        | 69.14 ( 84.92 )  | 54.45 ( 55.86 )  | 51.98 ( 52.64 )  | 58.18 ( 75.36 )  |
| DAAF-TriNet*   | 72.63 ( 87.17 )  | 55.01 ( 60.34 )  | 54.48 ( 58.71 )  | 60.12 ( 77.29 )  |
| [Bag of Tricks](https://github.com/michuanhaohao/reid-strong-baseline)   | 85.90 ( 94.50 )  | 60.90 ( 63.30 )  | 58.00 ( 59.10 )  | 76.40 ( 86.40 )  |
| [DAAF-BoT](https://github.com/CYFFF/DAAF-BoT)    | 87.90 ( 95.10 )  | 67.60 ( 69.00 )  | 63.10 ( 64.90 )  | 77.90 ( 87.90 )  |
| Bag of Tricks** | 94.20 ( 95.40 )  | 77.10 ( 74.40 )  | 73.40 ( 70.40 )  | 89.10 ( 90.30 )  |
| DAAF-BoT**      | 95.00 ( 96.40 )  | 82.00 ( 78.70 )  | 77.30 ( 73.90 )  | 89.60 ( 91.70 )  |

\* represents five crops and flip 

\*\* represents [re-ranking](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)







