# OVOS Model2Vec Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org), powered by [Model2Vec](https://github.com/MinishLab/model2vec) for fast, lightweight intent classification.

## Overview

This plugin integrates a pre-trained Model2Vec static embedding model with the OVOS intent pipeline system. It classifies natural language utterances into intent labels that have been registered at runtime by loaded skills (via Adapt, Padatious, or plugin-specific registration).

The pipeline works best as a fallback or confidence-based matcher. Use it when deterministic engines (Adapt, Padatious) fail to produce a high-confidence match.

## How It Works

1. An utterance arrives from the user.
2. The Model2Vec model runs inference and produces a probability distribution over all known intent labels.
3. The pipeline filters out any labels not currently registered by loaded skills.
4. The highest-probability valid intent is returned if it meets the configured confidence threshold.

## Key Properties

- **Pretrained, not adaptive.** The model was trained on a fixed corpus of OVOS skill intent examples. It cannot learn new skills at runtime.
- **Runtime filtering.** Even though the model knows hundreds of intent labels, only those from currently loaded skills are considered.
- **Tiered confidence.** Three match levels (`high`, `medium`, `low`) use separate thresholds, integrated into the OVOS pipeline priority system.
- **Multilingual by default.** The default model covers 11+ languages.

## Documentation

| Page | Description |
|------|-------------|
| [Installation](installation.md) | How to install the plugin |
| [Configuration](configuration.md) | All configuration options |
| [Pipeline Internals](pipeline.md) | Architecture and runtime behaviour |
| [Models](models.md) | Available pre-trained models and benchmark results |
| [Training](training.md) | How to build the dataset and train/retrain models |
| [Label scheme](labels.md) | Label format, families, dedup rules, and renames |
