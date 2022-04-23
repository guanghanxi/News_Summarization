Data Process:

1. data_divide.ipynb : Copy files from the original data folders to train data folder and test data folder

2. LSTM_data_process.ipynb: Read files in the train data folder and test data folder and generate train Dataloaders, test Dataloaders and vocab for the LSTM model

3. LED_data_process.ipynb: Read files in the train data folder and test data folder and generate train datasets and test datasets for the LED model

LSTM:

LSTM_trainer.ipynb: Set up or load a LSTM Seq2seq Model, train and save the model. It reads the three files generate by LSTM_data_process.ipynb

LED:

LED_trainer.ipynb: Set up or load a LEDConditionalGenerate Model, train and save the model.  It reads the two files generate by LED_data_process.ipynb

TEST:

test_baseline.py: Set up the evaluation metrics and two baseline models

LSTM_test.ipynb: Load a trained LSTM Seq2seq Model and evaluate it

LED_test.ipynb: Load a trained LED Model and evaluate it
