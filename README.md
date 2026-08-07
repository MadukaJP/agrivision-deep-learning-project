# AgriVision AI — Plant Disease & Crop Health Diagnostic SaaS

Agritech computer vision system that diagnoses crop diseases from leaf
photos, shows Grad-CAM heatmaps explaining the model's focus, and
generates a downloadable treatment recommendation PDF. Built for the
Deep Learning & Applied AI Engineering capstone.

## Architecture

```
[Cassava + PlantVillage-derived datasets] -> [dataset_loader.py: unified
manifest, stratified split, augmentation] -> [EfficientNetB3 transfer
learning: Phase 1 frozen backbone -> Phase 2 fine-tuned top layers] ->
[evaluate.py: confusion matrix, Grad-CAM, OOD test] -> [Flask app:
upload -> predict -> Grad-CAM overlay -> PDF report] -> [Hugging Face deployment]
```

- **Model**: EfficientNetB3 (ImageNet pretrained), two-phase transfer
  learning -- Phase 1 trains a dense classifier head on a frozen backbone,
  Phase 2 fine-tunes the top 40 backbone layers (with BatchNorm layers
  kept frozen -- see `DEFENSE_ANSWERS.md` for why) at a low learning rate
  with cosine decay.
- **Data**: 21 classes across 4 crops -- cassava (Kaggle competition
  dataset, real field photos), maize/tomato/pepper
  (`vipoooool/new-plant-diseases-dataset`, a PlantVillage derivative).
  Yam was scoped out -- no usable public labeled dataset exists for it.
- **Explainability**: Grad-CAM heatmaps generated per-prediction, shown
  alongside the diagnosis in the app.
- **Deployment**: Flask + Gunicorn on Hugging Face Spaces.

## Results

- Phase 1 (frozen backbone): 52.9% val accuracy, val_loss 1.552.
- Phase 2 (fine-tuned): 59.5% val accuracy, val_loss 1.316 (best at
  epoch 14 of 20; EarlyStopping restored these weights).
- **Test set** (held out, 5,835 images, never touched until final
  evaluation): 58% overall accuracy. Strong on maize/pepper (0.67-0.98
  F1), mixed on tomato, weakest on cassava's 4-way disease
  discrimination (0.26-0.58 F1) -- full breakdown and root-cause analysis
  in `DEFENSE_ANSWERS.md`.

See `DEFENSE_ANSWERS.md` for the full empirical evidence behind these
numbers, including two real bugs found and fixed during development
(a Grad-CAM preprocessing mismatch, and a BatchNorm-caused regression in
Phase 2 fine-tuning) and an honest account of where the OOD rejection
mechanism does and doesn't work.

## Repo structure

```
|-- config.py                 # paths, class mappings, hyperparameters
|-- data/dataset_loader.py    # manifest building, splitting, augmentation
|-- models/
|   |-- train.py               # two-phase training
|   |-- evaluate.py            # confusion matrix, Grad-CAM, OOD test
|   `-- saved_models/          # trained model + class_index.json
|-- app/
|   |-- app.py                 # Flask app (upload/predict/Grad-CAM/PDF)
|   |-- treatment_data.py      # disease -> treatment lookup table
|   |-- templates/, static/
|   `-- requirements.txt
|-- DEFENSE_ANSWERS.md
|-- BUSINESS_PLAN.md
`-- feature_log.md
```

## Running it

**Training** (Kaggle notebook, GPU): see cell-by-cell sequence in
`models/train.py` and `data/dataset_loader.py` -- datasets needed are
`cassava-leaf-disease-classification` (competition) and
`vipoooool/new-plant-diseases-dataset`.

**App, locally**:
```bash
cd app
pip install -r requirements.txt
python app.py
```
Requires `agrivision_final.keras` and `class_index.json` in
`models/saved_models/` (trained model, not included in this repo due to
size -- see Kaggle notebook output).

**Deployment**: Live app:
**[https://jpmaduka-agrivision.hf.space/](https://jpmaduka-agrivision.hf.space/)**.
