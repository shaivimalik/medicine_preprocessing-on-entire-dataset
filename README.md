## Reproducing "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures"

In this sequence of notebooks, we will reproduce the results from 

> M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.8936292.

which predicts pre-term birth based on raw EHG signals from pregnant women. It claims an accuracy of 95.5% on the test set using an SVM classifier with RBF kernel. We achieve a similar high accuracy (97%) by following the steps in that paper:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/Reproducing_Original_Result.ipynb) Reproducing "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures"

However, it turns out (as discussed in [2]) that the reported 95.5% accuracy is much higher than we would achieve on *new* EHG signals when the model is used in practice. This is because we oversampled the dataset before splitting it into training and test sets. Consequently, test set samples were used to generate synthetic samples for the training set and training set samples were used to generate synthetic samples for the test set.

We explore this issue further in the following examples - 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/Exploring_Oversampling-Adult.ipynb) Exploring Oversampling on the Income Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/Exploring_Oversampling-Synthetic.ipynb) Exploring Oversampling on Synthetic Data
 
Finally, we repeat the original pre-term birth prediction, but without the data leakage error - we keep training and test sets separate in oversampling, rather than oversampling all together. We show that the original accuracy was based on an "overly optimistic" evaluation, and the true performance of the model is much less.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shaivimalik/medicine_preprocessing-on-entire-dataset/blob/main/notebooks/Correcting_Original_Result.ipynb) "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures" Without Data Leakage

---

### Running the Project

#### Google Colab

Click on the "Open in Colab" buttons above to run the notebooks in Google Colab.

#### Chameleon

This resource may be executed on [Chameleon](https://chameleoncloud.org/). If using Chameleon, start by running the `reserve.ipynb` notebook in the Chameleon Jupyter environment.

#### Local Machine

1. Clone the repository:
   ```
   $ git clone https://github.com/shaivimalik/medicine_preprocessing-on-entire-dataset.git
   $ cd medicine_preprocessing-on-entire-dataset
   ```

2. Install the required dependencies:
   ```
   $ pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```
   $ jupyter notebook
   ```

---


### Acknowledgements

This project was part of the 2024 Summer of Reproducibility organized by the [UC Santa Cruz Open Source Program Office](https://ucsc-ospo.github.io/).

* Contributor: [Shaivi Malik](https://github.com/shaivimalik)
* Mentors: [Fraida Fund](https://github.com/ffund), [Mohamed Saeed](https://github.com/mohammed183)

---

### References

[1] M. U. Khan, S. Aziz, S. Ibraheem, A. Butt and H. Shahid, "Characterization of Term and Preterm Deliveries using Electrohysterograms Signatures," 2019 IEEE 10th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON), Vancouver, BC, Canada, 2019, pp. 0899-0905, doi: 10.1109/IEMCON.2019.8936292.

[2] Gilles Vandewiele, Isabelle Dehaene, György Kovács, Lucas Sterckx, Olivier Janssens, Femke Ongenae, Femke De Backere, Filip De Turck, Kristien Roelens, Johan Decruyenaere, Sofie Van Hoecke, Thomas Demeester, Overly optimistic prediction results on imbalanced data: a case study of flaws and benefits when applying over-sampling, Artificial Intelligence in Medicine, Volume 111, 2021, 101987, ISSN 0933-3657, https://doi.org/10.1016/j.artmed.2020.101987.