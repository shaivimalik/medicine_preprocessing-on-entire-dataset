## Reproducing "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures"

In this sequence of notebooks, we will reproduce a result from 

> M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.8936292.

which predicts pre-term birth based on raw EHG signals from pregnant women. It claims an accuracy of 95.5% on a test set using an SVM classifier with RBF kernel. We achieve a similar high accuracy (99%) by following the steps in that paper:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/01.ipynb) Reproducing "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures"

However, it turns out (as discussed in [2]) that the reported 95.5% accuracy is much higher than we would achieve on *new* EHG signals when the model is used in practice. This is because when we oversampled the data, we contaminated the validation set by using validation samples to generate "new" samples that ended up in the training set, and by using training samples to generate "new" samples that ended up in the validation set.

We explore this issue further in the following examples - 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/02.ipynb) Exploring Oversampling on the Income Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/03.ipynb) Exploring Oversampling on Synthetic Data
 
Finally, we repeat the original pre-term birth prediction, but without the data leakage error - we keep training and validation sets separate in oversampling, rather than oversampling all together. We show that the original accuracy was based on an "overly optimistic" evaluation, and the true performance of the model is much less.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/04.ipynb) "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures" Without Data Leakage


---

This resource may be executed on Google Colab or on [Chameleon](https://chameleoncloud.org/). The buttons above will open the materials on Colab. If you are using Chameleon, start by running the `reserve.ipynb` notebook inside the Chameleon Jupyter environment.

---


### Acknowledgements

This project was part of the 2024 Summer of Reproducibility organized by the [UC Santa Cruz Open Source Program Office](https://ucsc-ospo.github.io/).

* Contributor: [Shaivi Malik](https://github.com/shaivimalik)
* Mentors: [Fraida Fund](https://github.com/ffund), [Mohamed Saeed](https://github.com/mohammed183)

### References

[1] M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.8936292.

[2] Gilles Vandewiele, Isabelle Dehaene, György Kovács, Lucas Sterckx, Olivier Janssens, Femke Ongenae, Femke De Backere, Filip De Turck, Kristien Roelens, Johan Decruyenaere, Sofie Van Hoecke, Thomas Demeester, Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling, Artificial Intelligence in Medicine, Volume 111, 2021, 101987, ISSN 0933-3657, https://doi.org/10.1016/j.artmed.2020.101987.