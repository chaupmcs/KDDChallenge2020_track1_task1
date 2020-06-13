# KDD2020 - Multimodalities
KDD Cup 2020 Challenges for Modern E-Commerce Platform

### Link: https://tianchi.aliyun.com/competition/entrance/231786/rankingList
### The Challenge
- Training data : Search texts (queries) and their corresponding product images.
   + Note that the host of the challenge has run some objective detections on the raw images. Then they extract embedding vectors and bounding boxes for each object in the image. So, we are left with bunch of vectors, rather than normal images. They mentioned it's due to the copy right issues.
   + The data is from an E-Commerce website, with 3M rows in txt format (~ 124 GB).
- Requirements: Match the search text (query) and product image, i.e., for each query, pick top 5 most relevant images to the query => Learning Texts and Images => **Cross-modal retrieval** problem
  
  ![alt text](https://raw.githubusercontent.com/chaupmcs/KDDChallenge2020_track1_task1/master/query.png)
  
### Final Result
39th

### Some Related Papers
[1] [Deep Visual-Semantic Hashing for Cross-Modal Retrieval](https://www.kdd.org/kdd2016/papers/files/rpp0086-caoA.pdf)

[2] [Deep Supervised Cross-modal Retrieval](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)

[3] [Deep Cross-Modal Hashing](http://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf)

### Quick Start
```
python 3.7
tensorflow==1.15
Keras==2.2.4
keras-bert==0.78.0
```
