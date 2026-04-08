# OKGraph: Online Knowledge Graph Probing for Open-vocabulary Recognition
The official implementation of our paper "OKGraph: Online Knowledge Graph Probing for Open-vocabulary Recognition" (CVPR 2026 Findings)

## Overview
![Framework](assets/framework.png)

## abstract
Vision-language models (VLMs) have made significant strides in open-vocabulary recognition by aligning visual inputs with textual representations. However, most existing approaches treat categories as independent labels and rely on prompt tuning or feature adaptation, overlooking the underlying semantic structure that links related concepts. This lack of semantic connectivity limits the models’ ability to generalize, especially when the data distribution shifts or categories exhibit semantic ambiguity, where recognizing instances often depends on understanding their relationships to known concepts. We propose OKGraph (Online Knowledge Graph Probing), a dynamic framework that constructs, updates, and applies a semantic knowledge graph to guide recognition OKGraph integrates the perceptual capabilities of VLMs with the semantic reasoning of large language models (LLMs), enabling the adaptive extraction of contextual knowledge from test data. Through iterative contextual probing and model-guided correction, OKGraph refines its knowledge graph without requiring any training data or manual annotations. Extensive experiments across diverse vision tasks and domains demonstrate that OKGraph substantially improves zero-shot and few-shot performance, highlighting the value of structured reasoning in open-world vision.

The main contributions of this work are fourfold:

• We propose OKGraph, a training-free framework that enhances vision–language models for zero-shot recognition via dynamic semantic reasoning.

• We introduce a domain-aware contextual probing strategy that maintains a global scene graph and adaptively constructs goal-specific subgraphs based on prediction confidence and semantic diversity.

• We develop a self-correction mechanism that refines the knowledge graph through model feedback and blacklist-based forgetting to improve relevance and reduce noise.

• OKGraph achieves state-of-the-art performance across zero-shot and few-shot settings, demonstrating strong generalization under distribution shifts.


## Installation
```
# Clone this repo
git clone https://github.com/zenithc-git/OKGraph.git
cd OKGraph

# Create a conda enviroment
conda env create -f environment.yml
conda activate okgraph
```


## Data Format
Images are expected in an ImageFolder layout:
```
/path/to/images/
  class_0/
    img1.jpg
    ...
  class_1/
    img2.jpg
    ...
```

Global scene graphs are expected in JSON:
```
/path/to/base_knowledge/
  class_0/
    ..._names.json
    ..._names_addattris.json
  class_1/
    ..._names.json
    ..._names_addattris.json
  ...
```
We provide Global scene graphs for a subset of datasets in /path/to/base_knowledge/. For datasets not included in our release, the corresponding scene graphs can be constructed by following the format of the provided examples and leveraging GPT-3.5-Turbo or other large language models.

## How to Start

```
bash scripts/run.sh
```

## Reproducibility
- Deterministic seeds are set in the evaluation loop.
- Logs are written to `logs/` with timestamped filenames.
- Evaluation outputs include `entropy_log.json` and `prediction_log.json` for analysis.

## Results
## Zero-shot Classification

| Method | Caltech | Pets | Cars | Flowers | DTD | UCF101 | EuroSAT | Food101 | SUN397 | Aircraft | Easy | Hard | Avg |
|--------|--------|------|------|--------|-----|--------|---------|---------|--------|----------|------|------|------|
| CLIP | 91.5 | 87.3 | 65.2 | 66.3 | 44.6 | 63.9 | 43.0 | 83.5 | 62.1 | 23.1 | 88.4 | 76.5 | 66.3 |
| **CLIP + OKGraph** | **94.2 (+2.7)** | **88.6 (+1.3)** | **66.1 (+0.9)** | **72.5 (+6.2)** | **50.5 (+5.9)** | **68.7 (+4.8)** | 42.9 (-0.1) | **85.2 (+1.7)** | **68.0 (+5.9)** | **25.7 (+2.6)** | **89.9 (+1.5)** | **77.8 (+1.5)** | **69.2 (+2.9)** |
| TPT | 94.0 | 87.2 | 66.5 | 68.8 | 47.0 | 67.5 | 42.4 | 85.9 | 65.5 | 23.9 | 89.7 | 78.6 | 68.1 |
| **TPT + OKGraph** | **94.4 (+0.4)** | **88.6 (+1.4)** | **67.1 (+0.6)** | **71.3 (+2.5)** | **48.8 (+1.8)** | **69.9 (+2.4)** | **44.8 (+2.4)** | **86.4 (+0.5)** | **68.0 (+2.5)** | **25.0 (+1.1)** | **90.9 (+1.2)** | **79.7 (+1.1)** | **69.6 (+1.5)** |
| C-TPT | 93.4 | 88.8 | 66.3 | 68.3 | 46.2 | 64.6 | 43.9 | 85.2 | 65.1 | 24.4 | 88.9 | 77.0 | 67.7 |
| **C-TPT + OKGraph** | **94.0 (+0.6)** | **89.1 (+0.3)** | **66.6 (+0.3)** | **72.5 (+4.2)** | **50.4 (+4.2)** | **68.9 (+4.3)** | **44.8 (+0.9)** | **85.9 (+0.7)** | **67.5 (+2.4)** | **25.2 (+0.8)** | **89.1 (+0.2)** | **77.8 (+0.8)** | **69.3 (+1.6)** |
| TCA | 93.2 | 89.0 | 65.4 | 73.3 | 46.2 | 71.4 | 70.6 | 84.8 | 66.4 | 24.6 | 88.8 | 77.5 | 70.9 |
| **TCA + OKGraph** | **93.4 (+0.2)** | **89.2 (+0.2)** | **65.5 (+0.1)** | **74.1 (+0.8)** | **46.9 (+0.7)** | **72.0 (+0.6)** | **70.7 (+0.1)** | **84.9 (+0.1)** | **66.7 (+0.3)** | **24.8 (+0.2)** | **88.9 (+0.1)** | **77.8 (+0.3)** | **71.2 (+0.3)** |

## Few-shot Classification

| Method | Caltech | Pets | Cars | Flowers | DTD | UCF101 | EuroSAT | Food101 | SUN397 | Aircraft | Avg |
|--------|--------|------|------|--------|-----|--------|---------|---------|--------|----------|------|
| CLIP (a photo of a {}) | 91.2 | 86.8 | 60.4 | 66.0 | 42.9 | 60.9 | 43.9 | 83.2 | 62.1 | 20.2 | 61.8 |

### Hand-crafted Methods

| Method | Caltech | Pets | Cars | Flowers | DTD | UCF101 | EuroSAT | Food101 | SUN397 | Aircraft | Avg |
|--------|--------|------|------|--------|-----|--------|---------|---------|--------|----------|------|
| Template-80 | 91.6 | 86.8 | 60.4 | 66.1 | 42.8 | 62.9 | 52.6 | 84.2 | 63.5 | 19.5 | 63.0 |
| FILIP-8 | 91.4 | 87.4 | 60.7 | 67.0 | 43.4 | 65.0 | 54.3 | 84.6 | 63.9 | 18.9 | 63.7 |
| DEFILIP-6 | 91.0 | 87.5 | 59.9 | 66.5 | 41.3 | 63.6 | 46.4 | 84.3 | 62.3 | 18.8 | 62.2 |

### Description-based Methods

| Method | Caltech | Pets | Cars | Flowers | DTD | UCF101 | EuroSAT | Food101 | SUN397 | Aircraft | Avg |
|--------|--------|------|------|--------|-----|--------|---------|---------|--------|----------|------|
| DCLIP | 92.7 | 88.1 | 59.4 | 66.1 | 44.1 | 65.8 | 38.4 | 83.9 | 65.0 | 19.4 | 62.3 |
| Waffle | 92.1 | 87.7 | 59.3 | 66.3 | 43.2 | 64.5 | 51.6 | 84.9 | 65.0 | 19.6 | 63.4 |
| CuPL | 92.9 | 87.0 | 60.7 | 69.5 | 50.6 | 66.4 | 50.5 | 84.2 | 66.3 | 20.9 | 64.9 |
| GPT4Vis | 93.1 | 88.1 | 61.4 | 69.8 | 48.5 | 65.7 | 47.0 | 84.3 | 64.2 | 21.4 | 64.4 |
| AdaptCLIP | 92.7 | 87.6 | 59.7 | 67.2 | 47.4 | 66.5 | 51.3 | 84.2 | 66.1 | 20.8 | 64.4 |

| Method | Caltech | Pets | Cars | Flowers | DTD | UCF101 | EuroSAT | Food101 | SUN397 | Aircraft | Avg |
|--------|--------|------|------|--------|-----|--------|---------|---------|--------|----------|------|
| ProAPO w/ DCLIP | 93.1 | 90.1 | 61.3 | 73.6 | 49.7 | 66.1 | 57.6 | 84.2 | 65.9 | 22.6 | 66.4 |
| **ProAPO w/ DCLIP + OKGraph** | **93.2 (+0.1)** | **90.1 (0)** | **61.7 (+0.4)** | **74.0 (+0.4)** | **50.3 (+0.6)** | **67.6 (+1.5)** | **57.7 (+0.1)** | **84.9 (+0.7)** | **66.2 (+0.3)** | **23.2 (+0.6)** | **66.9 (+0.5)** |
| ProAPO | 93.9 | 90.2 | 61.6 | 74.7 | 53.3 | 68.1 | 57.6 | 84.8 | 66.4 | 22.7 | 67.3 |
| **ProAPO + OKGraph** | **94.0 (+0.1)** | **90.6 (+0.4)** | **62.0 (+0.4)** | **75.5 (+0.8)** | **53.6 (+0.3)** | **68.6 (+0.5)** | **57.6 (0)** | **84.9 (+0.1)** | **66.8 (+0.1)** | **23.1 (+0.4)** | **67.7 (+0.4)** |

## Natural Distribution Shift

| Method | IN-A | IN-V2 | IN-R | IN-S | Avg |
|--------|------|-------|------|------|------|
| CLIP | 47.9 | 60.6 | 73.8 | 46.0 | 57.1 |
| **CLIP + OKGraph** | **49.9 (+2.0)** | **62.6 (+2.0)** | **75.3 (+1.5)** | **47.8 (+1.8)** | **58.9 (+1.8)** |
| TPT | 52.6 | 63.1 | 76.2 | 47.8 | 59.9 |
| **TPT + OKGraph** | **53.4 (+0.8)** | **63.9 (+0.8)** | **77.0 (+0.8)** | **48.9 (+1.1)** | **60.8 (+0.9)** |
| C-TPT | 50.0 | 62.4 | 75.0 | 47.6 | 58.8 |
| **C-TPT + OKGraph** | **51.3 (+1.3)** | **63.3 (+0.9)** | **76.3 (+1.3)** | **48.6 (+1.0)** | **59.9 (+1.1)** |
| TCA | 49.3 | 61.8 | 75.7 | 48.6 | 58.9 |
| **TCA + OKGraph** | **49.6 (+0.3)** | **62.0 (+0.2)** | **75.8 (+0.1)** | **48.8 (+0.2)** | **59.1 (+0.2)** |



## Citation
If you use this code, please cite:
```
```

## License


## Acknowledgements

