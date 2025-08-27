# 🧠 Brain MRI Segmentation (U-Net Models)

This project demonstrates **Brain MRI image segmentation** using a U-Net model trained on MRI slices.  
It is packaged with **FastAPI** for the backend, **htmx** for interactivity, and deployed on **Render** using Docker.

---

## 🚀 Features
- **Preloaded Example Images** – choose from 5 sample MRI scans.
- **Multiple Model Weights** – pick from 3 different U-Net models (`V_1`, `epoch 9`, `epoch 20`) to see how segmentation improves with training.
- **Interactive Segmentation** – select an image + model, and instantly see the overlayed segmentation.
- **Stats Output** – segmentation percentages for background, gray matter, and white matter.
- **Reset Button** – easily go back and try another image.

---