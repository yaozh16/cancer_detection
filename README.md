# Data Description

This dataset includs the following files.

1. `train.tar.gz` includes the training data. After extracting this file, you can see a csv file named `train/feats.csv` that contains the clinical diagnostic data for each medical record. The ultrasound images of medical record `${id}` are under the folder `train/images/${id}`.
2. `test.tar.gz` includes the test data. You need to submit your predictions of all medical records in `test/feats.csv`.
3. `submission_sample.csv` is a sample submission file in the correct format. The file should have N lines where N is the number of records in `test/feats.csv`. Each line contains a patient ID and a prediction seperated by comma.
4. `evaluate.py` is used for evaluation. You can get the Accuracy and Macro-F1 score by running:

```bash
python evaluate.py ${truth_path} ${prediction_path}
```

If you have any problems, please contact cyk18@mails.tsinghua.edu.cn.