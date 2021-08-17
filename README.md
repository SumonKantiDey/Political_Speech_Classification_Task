# Political Speech Classification Task
The goal of this project is to finetune a transformer model that identifies political speech talks about immigration or not. Here we have used a standalone RoBERTa base transformer model. We did not perform robust preprocessing except lightweight preprocessing [[NoteBook]](https://github.com/SumonKantiDey/Political_Speech_Classification_Task/blob/main/preprocess/preprocessing.ipynb). We have tried standalone RoBERTa baseline with 200 maximum sequence lengths (required F1 score mention in the results section)[[NoteBook]](https://github.com/SumonKantiDey/Political_Speech_Classification_Task/blob/main/training_without_spliting_data.ipynb).  Recurrence over BERT (Pappagari, Raghavendra, et al) which is used long document classification task, where split the text into fixed-size segments. We have implemented RoBERT (Recurrence over BERT) technique with standalone RoBERTa where the F1 score and Matthews correlation coefficient (MCC) score improved over the standalone RoBERTa with 200 maximum sequence lengths.[[NoteBook]](https://github.com/SumonKantiDey/Political_Speech_Classification_Task/blob/main/training_with_spliting_data.ipynb).

#### Folder structure ####	
~~~
data_merge.py       --> Used to merge the dataset based on doc id.
./input             --> Contains the dataset related files.
./preprocess  	    --> Contains the codes for preprocessing the dataset.	
./results           --> Contain the evaluation result based on test dataset.
./src               --> Contains the codes for all transformer-based classifiers.
~~~
#### Dataset ####
I have got two dataset dat_speeches1 contains **42540** samples and dat_speeches2 contains **34354** samples. From those data total label data distribution given below.
| type  | # of examples | 
| ---       |---     |
|immigration| 1126|
|not immigration| 874|
| Total | 2000
Then, we split the samples into train (1600), validation (200), test (200) sets.

Total unseen samples without label is **74894**.

### Code usage instructions ### 
First clone this repo and move to the directory. Then, install necessary libraries. Also, following commands can be used: 
~~~
$ git clone https://github.com/SumonKantiDey/Political_Speech_Classification_Task.git
$ cd Political_Speech_Classification_Task/ 
$ pip install -r requirements.txt
~~~

### Parameters ####
```
Learning-rate = 2e-5,3e-5]
Epochs = [2,4,5]
Max seq length = [128 192 200]
Dropout = [0.1 0.2 0.3]
Batch size = [8,16]
```

### Results ###
Standalone RoBERTa-base result base on test data.
Please note the following class encoding to interprete the class-specific classification reports:

- immigration : class 1
- not immigration : class 0

standalone RoBERTa without segmentation:

```
Mcc Score:: 0.5709861004252481
Accuracy:: 0.79
Precision:: 0.7872515703841005
Recall:: 0.783745295493846
F_score:: 0.7851662404092071
classification_report::               
                precision    recall  f1-score   support
         0.0       0.77      0.74      0.75        87
         1.0       0.80      0.83      0.82       113

    accuracy                           0.79       200
   macro avg       0.79      0.78      0.79       200
weighted avg       0.79      0.79      0.79       200
```

standalone RoBERTa with segmentation:
```
Mcc Score:: 0.6007749117509227
Accuracy:: 0.795
Precision:: 0.7980769230769231
Recall:: 0.8027158986878242
F_score:: 0.7945840326661489
classification_report::               
                precision    recall  f1-score   support

         0.0       0.72      0.86      0.79        87
         1.0       0.88      0.74      0.80       113

    accuracy                           0.80       200
   macro avg       0.80      0.80      0.79       200
weighted avg       0.81      0.80      0.80       200

```

# Reference
1. Pappagari, Raghavendra, et al. "Hierarchical transformers for long document classification." 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2019.
2. Ding, Ming, et al. "Cogltx: Applying bert to long texts." Advances in Neural Information Processing Systems 33 (2020): 12792-12804.
3. Chicco, Davide, and Giuseppe Jurman. "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation." BMC genomics 21.1 (2020): 1-13.

#  <span style="color: red"> THE WORK IS IN PROGRESS </span>
