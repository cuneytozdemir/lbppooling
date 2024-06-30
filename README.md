### About

This repository contains the code for the paper titled "A new local pooling approach for convolutional neural network: local binary pattern," published in *Multimedia Tools and Applications* (Ozdemir, C., Dogan, Y., & Kaya, Y., 2024). The paper proposes an LBP-based pooling method that reduces information loss by considering the neighborhood and size of pixels in the pooling region. This method improves performance on multiple datasets compared to traditional max and average pooling methods.

**Abstract:** The pooling layer in CNN models aims to reduce the resolution of image/feature maps while retaining their distinctive information, reducing computation time, and enabling deeper models. Max and average pooling methods are frequently used due to their computational efficiency; however, they discard the position information of the pixels. In this study, we proposed an LBP-based pooling method that generates a neighborhood-based output for any pixel, reflecting the correlation between pixels in the local area. Our proposed method reduces information loss by considering the neighborhood and size of the pixels in the pooling region. Experimental studies on four public datasets demonstrated the effectiveness of the LBP pooling method. Improvements of 1.56% for Fashion MNIST, 0.22% for MNIST, 3.95% for CIFAR10, and 5% for CIFAR100 were achieved using a toy model. Performance improvements for CIFAR10 and CIFAR100 were also observed with transfer learning models. The proposed method outperforms the commonly used pooling layers in CNN models.

### Citation
If you find this work or the code in this repository useful in your research or projects, please consider citing our paper:

Ozdemir, C., Dogan, Y., & Kaya, Y. (2024). A new local pooling approach for convolutional neural network: local binary pattern. *Multimedia Tools and Applications, 83*(12), 34137-34151. [https://doi.org/10.1007/s11042-023-17540-x](https://link.springer.com/article/10.1007/s11042-023-17540-x)

@article{ozdemir2024local,
  title={A new local pooling approach for convolutional neural network: local binary pattern},
  author={Ozdemir, C. and Dogan, Y. and Kaya, Y.},
  journal={Multimedia Tools and Applications},
  volume={83},
  number={12},
  pages={34137--34151},
  year={2024},
  publisher={Springer}
}

### Acknowledgment

Please ensure that proper attribution is given when using or referencing this work. If you utilize the methods or code provided in this repository, kindly include a citation to our paper as described above.
