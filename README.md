# Deep Morphological Profiling of Immune Cells in Peripheral Blood of Cancer Patients

## Project Overview

The initiation and progression of cancer are intricately linked to the interplay between the tumor and the immune system, both locally in the tumor tissue and systemically in peripheral blood. Immunoediting, a process where cancer suppresses anti-tumor immunity, unfolds through distinct phases—from the immune system's elimination of abnormal and potentially cancerous cells in the early stages to the eventual immune escape of tumor cells as the disease progresses.

Recent evidence suggests that circulating tumor cells (CTCs) form clusters with white blood cells (WBCs) in cancer patients, representing potential instances of tumor-immune interaction. However, the underlying biological mechanisms and clinical implications of these interactions remain largely unknown.

### Challenges

CTCs are rare in cancer patients, making their identification challenging. To address this, we previously developed the high-definition single cell assay (HDSCA)—a platform that enables the identification and study of CTCs, CTC clusters, and other circulating rare events among millions of WBCs. The HDSCA involves collecting immunofluorescent (IF) microscopy image data from approximately 3 million cells. Our goal is to leverage deep learning tools to deconvolve the immune repertoire from IF image data and potentially predict immune cell subclasses.

## Project Goals

We aim to:
- Perform morphological characterization of immune cells from IF microscopy images derived from peripheral blood samples of cancer patients.
- Utilize deep learning and computer vision techniques to analyze instances of tumor-immune interaction in peripheral blood.
- Develop state-of-the-art deep learning models to address critical questions in the field of liquid biopsy.
- Learn latent feature space to investigate morphologically distinct  immune cell populations.

## Cell Classification Tasks
- Binary classification: Lymphocytes (T-cells, B-cells, NK-cells) vs Myelocytes (Granulocytes, monocytes/macrophages).
- 5-classes classification: T-cells, B-cells, NK-cells, Granulocytes, monocyte/macrophages.
- 6-classes classification: CD4+ T-cells, CD8+ T-cells, B-cells, NK-cells, Granulocytes, monocyte/macrophages.

## Model Performances
![image](https://github.com/user-attachments/assets/768badbe-2f9f-474f-9832-f80e35d8d2a3)
![image](https://github.com/user-attachments/assets/4dfb0ed6-484c-4149-ba80-bad4d83ad175)
![image](https://github.com/user-attachments/assets/98c684d1-bd4e-479c-8cac-0a5306e85c6c)

## Repository Structure

- **`data_preprocessing/`**: Contains all essential steps to generate IF images and corresponding labels for training and testing.
- **`classification_training/`**: Includes CNN model classifiers with training codes and pre-trained models, including Lymphocytes/Myelocytes classifier, Lymphocytes subclassifier, Myelocytes subclassifier, multi-class classifier. 

## Contact

If you are interested in our research or have any questions, please reach out to me at sl5632@columbia.edu
