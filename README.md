# Medical Name Entity Recognition using Hybrid LSTM and Rule Based Technique


## Introduction

Recognizing Named Entities in medical records is not same as other Name Entities recognition tasks because medical records have their own format of storing information about patient’s disease, and previous diagnosis. Wording used in medical records is a lot different from other text documents and many abbreviations such as CXR, PA, mg and alot of chemical notations are used which are difficult to understand even for non professionals. The publicly available data for medical NER is quite low as compared to other NER datasets. Since data is low we cannot use any complex network because it would overfit and we will not be able to make relaible inference. We cannot use transfer learning because there is a lot of difference between medical records and other text documents. So we have to design a network by carefully selecting it’s parameter so that network learns a generalized pattern from training data. 

## Architecture

Pipeline consists of a FastText model [1] with the embedding vector of size 300 and moving window of length 15. It is used to incorporate unknown words during inference. Main deep learning network includes a tokenization layer, an embedding layer with embedding size of 450, a Bi directional LSTM with 150 units and a time distributed Dense layer of 100 units. Finally a CRF layer with 3 unit to predict medicine, non medicine, and medicine related word. Network is optimised with CRF loss. After this we performed False positive reduction using a list of common non medical English words and compared predicted annotation with this list.

![NER](https://github.com/Azkarehman/NLP-for-Medical-NET-Recognition/blob/main/im.png)
