# Multi-class Classification

## 1. Objective

- Neural Network Architecture (PyTorch)
  - Convolutional layer
  - Stride, Padding
  - Pooling
  - Batch Normalization
- Optimization (PyTorch)
  - optim.SGD (zero_grad, step)
  - multi-class loss (BCELoss, BCEWithLogitsLoss, one-hot encoding)

## 2. Baseline notebook code

- [assignment_02.ipynb](assignment_02.ipynb)

## 3. Utility codes to complete

- [mydataset.py](mydataset.py)
- [mymodel.py](mymodel.py)
- [mytrain.py](mytrain.py)
- [myinfo.py](myinfo.py)

## 3. Data

- mnist
  - train: class 0, 1, 2, 3, 4
  - test: class 0, 1, 2, 3, 4

## 4. Neural Network

- convolutional layer (stride, padding)
- linear layer (Flatten)
- pooling
- batch normalization
- activation function (Sigmoid, ReLU)

## 5. Loss function

- `torch.nn.BCEWithLogitsLoss`
- `torch.nn.BCELoss`

## 6. Gradient

- `loss.backward()`

## 7. Optimization by Stochastic Gradient Descent

- `torch.optim.Optimizer.zero_grad()`
- `torch.optim.Optimizer.step()`

## 8. Configuration

- batch size
- learning rate
- number of epoch
- initialization of model parameters
- neural network architecture:
  - convolutional layer
  - linear layer
  - pooling
  - activation
  - batch normalization
- optimization:
  - back-propagation
  - stochastic gradient descent
- loss:
  - BCELoss, BCEWithLogitsLoss

## 9. GitHub history

- `commit` should be made at the beginning for each python file
- `commit` should be made at the end for each python file
- `commit` should be made at least 10 times in between for each file 
- `commit` message is recommended to be meaningful
- `commit` history is desirable to effectively indicate the progress of the coding

---

## Submission (Google Classroom)

1. jupyter notebook in `ipynb` format
2. jupyter notebook in `pdf` format
3. completed utility codes in `py` format
4. GitHub history for each utility python code in `pdf` format
5. result output in `csv` format
