# RL-SynthRS

##  Quick Start

### 1. Install Dependencies

Make sure you are using Python 3.8 or later. Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Add Real Images

Place your real remote sensing images (e.g., `.jpg`, `.png`) into the `test_real/` directory. These images are used as reference for training.

### 3. Train the Model (Reinforcement Learning)

Run the following command to start the training process:

```bash
python train.py
```

After training, a configuration `.txt` file will be saved in the `config/` directory.

### 4. Generate Synthetic Data

Use the generated configuration file to synthesize images:

```bash
python renderer.py --config config/your_config_file.txt
```

The generated synthetic images will be saved automatically for use in further training or evaluation of object detection models.

---
