## face emotion recgonition competition

the training dataset is splitted into 7 categories, 

|label|emotion|n_samples|
|---|------|------------|
|0|angry   | 514        |
|1|disgust | 558        |
|2|fear    | 194        |
|3|happy   | 1759       |
|4|neutral | 1617       |
|5|sad     | 1154       |
|6|surprise| 1160       |

data: https://pan.baidu.com/s/1nuXSgxb

* data-train.tar.gz: training data (6956 images)
* data-test.tar.gz: testing data (3040 images)
* expression_test_result.csv (true labels for test data)

* requirements
 * opencv `conda install opencv -c menpo`
 * keras
 * imageio `conda install -c conda-forge imageio`
 
