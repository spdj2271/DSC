# DSC
## Paper
Deep Spectral Clustering via Simultaneous Spectral Embedding and Entropy Minimization (DSC)

## Usage
1) Clone the code locally.
```
git clone https://github.com/spdj2271/DSC.git DSC
```
2) Launch an experiment on **USPS** dataset.

```
cd DSC
python main.py
```

3)  Launch an experiment on other dataset, e.g., 'FASHION', 'MNIST', 'MNIST_test', 'FASHION_test', 'FRGC', 'COIL20'. 
```
python main.py --dataset=FASHION
```


## Dependencies
tensorflow 2.4.1

scikit-learn 0.23.2

numpy 1.19.5

scipy 1.2.1
