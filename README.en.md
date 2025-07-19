> 🌐 This page is also available in [Russian 🇷🇺](./README.md)

# Pyolimp

This project provides a machine learning-based framework for
image precompensation targeting vision defects,
specifically color vision deficiencies (CVD) and
refractive visual impairments (RVI).
The framework incorporates both neural network and non-neural network modules
to address precompensation effectively across different approaches.

## Project Overview

This project focuses on precompensating for visual
impairments using machine learning techniques.
It includes a comprehensive framework that supports both neural network (NN)
and non-neural network (non-NN) methods to restore image quality for those affected by:

* Color Vision Deficiencies (CVD):
Compensates for color blindness (e.g., protanopia, deuteranopia).
* Refractive Visual Impairments (RVI):
Addresses distortions caused by refractive errors, improving clarity and sharpness.
* Another Visual Impairments:
This project could be used for precompensation different visual impairments, but, for example, this project provides only methods for CVD and RVI precompensation.

## Requirements

* Python 3.10+
* Pytorch 2.4+
* Additional dependencies listed in pyproject.toml

PC Technical Specifications:
* a hardware platform with OS Ubuntu 64-bit version 22.04 LTS or higher or Windows version 10 or higher;
* at least 16 GB RAM;
* 8-core processor x86 2.5 GHz or higher or similar one;
* GPU with cuda 12.+ support, and at least 12 GB of video memory;
* at least 100 GB of free hard disk space;
* keyboard and mouse (or touchpad).

## Installation

```
pip install olimp
```
or
```
pip install git+https://github.com/pyolimp/pyolimp.git
```

## Usage

### 1. Non-Neural Network Modules for CVD and RVI Precompensation

To run the RVI optimization algorithm for precompensation using the
Bregman-Jumbo method, execute:
```
python3 -m olimp.precompensation.optimization.bregman_jumbo
```

To run the RVI optimization algorithm for precompensation using the
Montalto method, execute:
```
python3 -m olimp.precompensation.optimization.montalto
```
To run the CVD optimization algorithm for precompensation using the
Tennenholtz Zachevsky method, execute:

```
python3 -m olimp.precompensation.optimization.tennenholtz_zachevsky
```

You can also call examples from the directory `olimp.precompensation.basic`
and `olimp.precompensation.analytics` as in the examples given.

### 2. Neural Network Modules for CVD and RVI Precompensation

To run the NN model for RVI precompensation using the USRNET method, execute:
```
python3 -m olimp.precompensation.nn.models.usrnet
```

To run the NN model for CVD precompensation based on the Swin transformers, execute:
```
python3 -m olimp.precompensation.nn.models.cvd_swin.Generator_transformer_pathch4_844_48_3_nouplayer_server5
```

### 3. Training models

To train neural network models for precompensation, use the following command:

```
python3 -m olimp.precompensation.nn.train.train --config ./olimp/precompensation/nn/pipeline/usrnet.json
```
you can also train other models, please see `olimp/precompensation/nn/pipeline`. Also we have **json schema** and you can generate it, use the following command:

```
python3 -m olimp.precompensation.nn.train.train --update-schema
```

### Examples
#### CVD demo example
<img src="https://github.com/user-attachments/assets/42f54054-dba1-4204-957e-29b1a44a690c">

#### RVI demo example
<img src="https://github.com/user-attachments/assets/7e35fe3b-7667-4530-8c79-a1263749eeff">