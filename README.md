# Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques.

This repository contains the code for the paper of the same name, which introduces the Electric Guitar Instrumental Playing Techniques (EG-IPT) dataset and the `ipt~` Max/MSP external object.

This work was presented at the [NIME 2025](https://nime2025.org/) conference in Canberra, Australia.

## 📁 Project Structure

```
nime2025/
├── data/
│   ├── raw/                 # Raw audio recordings
│   └── dataset/             # Store dataset csv files
├── augments.py              # Data augmentation definitions
├── externals/               # Batch sampler implementation
├── model/                   # Model definitions
├── utils.py                 # Dataset management, audio processing, and training functions
├── nime2025.ipynb           # Main notebook
└── requirements.txt         # Project dependencies
```

## 🚀 Getting Started

### 1. Download and prepare the dataset

Download the EG-IPT dataset from [here](to be completed).  
Once downloaded, extract it and place the contents `data/raw/` directory of this repository.

Make sure path point to correct location of the dataset.
The directory structure should look like this:
```
data/raw/EG-IPT/pickups/
├── HB-neck/
├── HB-bridge/
└── HB-couple/
```

Paths can be modified in the notebook if needed.

### 2. Python environment

Make sure you have **Python 3.11.11** installed. We recommend using a dedicated conda environment:

```bash
conda create --name nime2025 python=3.11.11
conda activate nime2025
```

### 3. Run the notebook

Open `nime2025.ipynb` in Jupyter and run all cells sequentially. It will guide you through:

The notebook automates all steps, including:
- Installing required Python packages
- Verifying dataset structure
- Splitting data and performing preprocessing
- Training the model and evaluating it
- Exporting a TorchScript `.ts` model for real-time usage

## 🎛️ Real-time Deployment in Max

For real-time use in Max, check our external object available here:  
👉 [ipt_tilde](https://github.com/nbrochec/ipt_tilde)

This repository provides the code necessary to compile a Max object capable of loading `.ts` models exported from the jupyter notebook `nime2025.ipynb`.

## 🧠 About

This project is part of an ongoing research effort into the real-time recognition of instrumental playing techniques for interactive music systems.
If you use this work in your paper, please consider citing the following:

```bibtex
@inproceedings{fiorini2025egipt,
  title={Introducing EG-IPT and ipt~: a novel electric guitar dataset and a new Max/MSP object for real-time classification of instrumental playing techniques},
  author={Fiorini, Marco, and Brochec, Nicolas and Borg, Joakim and Pasini, Riccardo},
  booktitle={NIME 2025},
  year={2025},
  address={Canberra, Australia}
}
```

## 📚 Related Work

If you are interested in this topic, please check out our other papers:
- [Fiorini and Brochec (2024)](https://hal.science/hal-04635907) - "Guiding Co-Creative Musical Agents through Real-Time Flute Instrumental Playing Technique Recognition"
- [Brochec et al. (2024)](https://hal.science/hal-04642673) - "Microphone-based Data Augmentation for Automatic Recognition of Instrumental Playing Techniques"
- [Brochec and Tanaka (2023)](https://hal.science/hal-04263718) - "Toward Real-Time Recognition of Instrumental Playing Techniques for Mixed Music: A Preliminary Analysis"


## 📜 License

This project is released under a GPL-3.0 license.
