1.each image has been normalized between [-1,1]
2.image shape [W,H,C] e.g., [28,28,1]
3.Full datasets can be downloaded [here](https://drive.google.com/drive/folders/1bLVnwJxx8uN8jujq7MJl1Nydd6FaCmFT?usp=sharing). 




name               sample #     cluster #    dimension
COIL-20             1,440           20              28,28,1 (resized)
USPS                  9,298           10             16,16,1
FRGC                  2,462           20              32,32,3
MNIST/FASHION         7,0000        10              28,28,1   			




```
# loading dataset code
import numpy as np
import h5py
with h5py.File(f"ds/{name}.h5", "r") as f:
    x = np.array(f['x'][:])
    y = np.array(f['y'][:])
```

	


