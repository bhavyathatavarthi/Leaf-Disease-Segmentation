# Leaf Disease Segmentation
This project performs **binary leaf disease segmentation** using a **U-Net model (ResNet18 encoder)** trained in PyTorch.  
The model predicts infected regions on leaf images using pixel-wise segmentation.
## Dataset
This dataset is from kaggle it has 588 images and masks and augumented data of 2940 images and masks [Dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset)
* Images → RGB .jpg /.png
* Masks → Grayscale .png
### Mask pixels must contain:
* 0 → Background
* 1 → Diseased region
## Model Architecture
We used:
- **U-Net** from `segmentation_models_pytorch`
- Encoder: `resnet34`
- Loss: **0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss**
- Optimizer: **Adam**
- Learning Rate: `1e-3`
## Training Pipeline
1. Load images and masks using custom `LeafDataset` class.
2. Apply transforms:
   - Resize
   - Normalize  
3. Masks are:
   - Converted to NumPy  
   - Thresholded  
   - Converted to tensor and reshaped to `[1, H, W]`
4. Model trained for 10 epochs.
5. Outputs converted using:
   ```python
   preds = torch.sigmoid(output)
   preds = (preds > 0.3).float()
## 📊 Evaluation Metrics
To evaluate the performance of the segmentation model, the following metrics were used:
### **1️⃣ Dice Coefficient (F1 Score)**
The Dice score measures the overlap between the predicted mask and the ground truth mask.
- Ranges from **0 to 1**
- I got Dice Coefficient of 0.83
### **2️⃣ Intersection over Union (IoU / Jaccard Index)**  
IoU measures how much the prediction overlaps the ground truth compared to their union.
- Ranges from **0 to 1**
- I got IoU score of 0.75
### **3️⃣Loss Function = Binary Cross Entropy (BCE) Loss + Dice Loss**
- I got 0.10 loss 

