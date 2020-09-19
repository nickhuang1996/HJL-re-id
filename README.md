# HJL-re-id
A awesome project for person re-id.

# Introduction
- This repository is for person re-id including supervised learning, unsupervised learning and occluded person re-id task. You can utilize this code to learn how to make `person re-id`tasks. 
- For HJL, this is my name initials -- Huang Jialiang.
- I has estabilished the structure containing the introduction of person re-id models like `PCB`,`MGN` and `MDRS`(my paper), `PGFA` and `HOReID`. These structures are all reproduced by myself in this code framework.
- If you have any quesions, please contact me by my email. 
- My emails is: nickhuang1996@126.com.

# Performances
## Market1501
| Methods | mAP |	Rank-1 | Rank-5 |	Rank-10 | 
|---|---|---|---|---|
| MDRS |	87.6 | 95.8 |	98.4 | 99.1 |
| Pyramid | 88.2 | 95.7 | 98.4 | 99.0 |
| DSA-reID | 87.6 | 95.7	| – |	– |
| MGN | 86.9 | 95.7 | – | – |
| PCB+triplet | 83.0 | 93.4 | 97.8 | 98.4 |
| CASN(PCB) | 82.8 | 94.4 | – | – |
| PCB+RPP | 81.6 | 93.8 | 97.5 | 98.5 |
| VPM | 80.8 | 93.0 | 97.8 | 98.8 | 
| PCB | 77.4 | 92.3 | 97.2 | 98.2 | 
| GLAD | 73.9 | 89.9 | – | – |
| MultiScale | 73.1 | 88.9 | – | – |
| PartLoss | 69.3 | 88.2 | – | – |
| PDC | 63.4 | 84.4 | – | – |
| MultiLoss | 64.4 | 83.9 | – | – |
| PAR | 63.4 | 81.0 | 92.0 | 94.7 |
| HydraPlus | – | 76.9 | 91.3 | 94.5 |
| MultiRegion | 41.2 | 66.4 | 85.0 | 90.2 |
| SPReID |	83.4 | 93.7 | 97.6 | 98.4 |
| AOS | 70.4 | 86.5 | – | – |
| Triplet Loss | 69.1 | 84.9 | 94.2 | – |
| Transfer | 65.5 | 83.7 | – | – |
| PAN | 63.4 | 82.8 | – | – |
| SVDNet | 62.1 | 82.3 | 92.3 | 95.2 |

## DukeMTMC-reID
| Methods | mAP |	Rank-1 | Rank-5 |	Rank-10 | 
|---|---|---|---|---|
| MDRS | 79.4 | 89.4 | 95.1 | 96.8 |
| Pyramid | 79.0 | 89.0 | 94.7 | 96.3 |
| MGN | 78.4 | 88.7 | – | – |
| CASN(PCB) | 73.7 | 87.7 | – | – |
| DSA-reID | 74.3 | 86.2 | – | – |
| SPReID | 73.3 | 86.0 | 93.0 | 94.5 |
| PCB+triplet | 73.2 | 84.1 | 92.4 | 94.5 |
| VPM | 72.6 | 83.6 | 91.7 | 94.2 |
| PCB+RPP | 69.2 | 83.3 | – | – |
| PSE+ECN | 75.7 | 84.5 | – | – |
| DNN + CRF | 69.5 | 84.9 | – | – |
| GP-reid | 72.8 | 85.2 | – | – |
| AOS | 62.1 | 79.2 | – | – |


