# Political_Speech_Classification_Task

#### Folder structure ####	
~~~
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


# Reference
1. Pappagari, Raghavendra, et al. "Hierarchical transformers for long document classification." 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2019.
2. Ding, Ming, et al. "Cogltx: Applying bert to long texts." Advances in Neural Information Processing Systems 33 (2020): 12792-12804.
