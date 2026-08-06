# YUPU: A Benchmark and Multimodal Diffusion Model for ALS Point Cloud Upsampling

Official repository for our paper accepted at an **ECCV 2026 Workshop**.

> **Code release:** Coming soon
> **Dataset release:** Coming soon

[Paper](PAPER_URL) | [Project Page](PROJECT_PAGE_URL) | [Dataset](DATASET_URL)

<p align="center">
  <img src="assets/overview.png" width="900" alt="Overview of YUPU and GCDM">
</p>

## Overview

Airborne laser scanning (ALS) point clouds are valuable for large-scale 3D scene understanding, but collecting dense and complete scans is expensive and operationally demanding. Point-cloud upsampling provides a practical way to reconstruct denser scenes from sparse observations.

Most existing upsampling benchmarks generate sparse inputs by synthetically decimating dense point clouds. However, this process does not fully reproduce the irregular sampling patterns, occlusions, and local density variations found in actual ALS acquisitions.

To address this limitation, we introduce:

* **YUPU**, a scene-level ALS point-cloud upsampling benchmark constructed from overlapping physical flight-line acquisitions.
* **GCDM**, a Geometry-Appearance Conditioned Diffusion Model that combines point coordinates, local surface geometry, and multi-view projection features for point-cloud reconstruction.

## News

* **[2026]** The paper was accepted at an ECCV 2026 Workshop.
* **Code and pretrained models are coming soon.**
* **YUPU dataset access instructions are coming soon.**

## YUPU Dataset

YUPU provides realistic sparse–dense point-cloud pairs created from separately acquired and spatially aligned ALS flight lines.

Unlike conventional synthetic benchmarks, the sparse input is not produced solely by randomly decimating its dense reference. Instead:

* A sparse sample is obtained from an individual flight-line acquisition.
* Its dense reference is constructed by aggregating overlapping physical acquisitions.
* Training, validation, and test samples are separated geographically to prevent spatial data leakage.

### Dataset characteristics

| Property           | YUPU                                                             |
| ------------------ | ---------------------------------------------------------------- |
| Domain             | Airborne laser scanning                                          |
| Task               | Scene-level point-cloud upsampling                               |
| Sparse input       | Individual physical acquisition                                  |
| Dense reference    | Aggregated overlapping acquisitions                              |
| Upsampling factors | ×2, ×3, and ×4                                                   |
| Input size         | 12,000 points                                                    |
| Target sizes       | 24,000, 36,000, and 48,000 points                                |
| Number of areas    | 13                                                               |
| Number of samples  | 1,300                                                            |
| Split strategy     | Region-disjoint                                                  |
| Scene types        | Buildings, vegetation, roads, plazas, and parking infrastructure |

<p align="center">
  <img src="assets/dataset_examples.png" width="900" alt="Representative scenes from the YUPU dataset">
</p>

## Method

Our **Geometry-Appearance Conditioned Diffusion Model (GCDM)** extends diffusion-based point-cloud reconstruction with three complementary conditions:

1. **Point condition**
   Encodes the spatial structure of the sparse input point cloud.

2. **Geometry condition**
   Uses surface normals estimated through online PCA-kNN to provide explicit local geometric information.

3. **Projection-based condition**
   Renders the sparse point cloud from ten virtual viewpoints and extracts contextual features from the resulting projections.

The three conditions are integrated through a lightweight cross-modal attention module and used to guide the diffusion-based reconstruction process.

<p align="center">
  <img src="assets/method.png" width="900" alt="Architecture of the proposed GCDM">
</p>

## Quantitative Results

The following results summarize performance on the YUPU ×4 test set.

| Method          |       CD ↓ |   JSD-3D ↓ |      F1 ↑ |      EMD ↓ |
| --------------- | ---------: | ---------: | --------: | ---------: |
| PUDM            |     1.2277 |          — |     63.85 |     7.1298 |
| LiDiff          |     0.6057 |     0.7029 |     72.49 |     3.0006 |
| **GCDM (Ours)** | **0.4842** | **0.6712** | **84.06** | **2.4721** |

GCDM improves Chamfer Distance, F1 score, and Earth Mover’s Distance over the evaluated diffusion-based baselines, demonstrating the benefit of combining spatial, geometric, and projection-based conditions.

## Qualitative Results

<p align="center">
  <img src="assets/qualitative_results.png" width="900" alt="Qualitative comparison of point-cloud upsampling methods">
</p>

From left to right: sparse input, PUDM, LiDiff, GCDM, and dense reference.

## Installation

Installation instructions will be provided with the source-code release.

```bash
git clone https://github.com/[USERNAME]/[REPOSITORY].git
cd [REPOSITORY]
```

## Dataset Preparation

Dataset download and preprocessing instructions are coming soon.

The planned release will include:

* Training, validation, and test splits
* Sparse–dense pairs for ×2, ×3, and ×4 upsampling
* Dataset metadata
* Preprocessing scripts
* Evaluation scripts

## Training and Evaluation

Training configurations, pretrained checkpoints, and evaluation commands will be added with the code release.

```bash
# Training command — coming soon
python train.py [CONFIGURATION]

# Evaluation command — coming soon
python test.py [CONFIGURATION] [CHECKPOINT]
```

## Release Checklist

* [ ] YUPU dataset
* [ ] Dataset preprocessing code
* [ ] GCDM training code
* [ ] Evaluation code
* [ ] Pretrained models
* [ ] Configuration files
* [ ] Additional qualitative results

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{yoo2026yupu,
  title     = {YUPU: A Benchmark and Multimodal Diffusion Model for ALS Point Cloud Upsampling},
  author    = {[Author names]},
  booktitle = {Proceedings of the European Conference on Computer Vision Workshops},
  year      = {2026}
}
```

The final BibTeX entry will be updated after publication.

## Acknowledgements

Our implementation builds upon ideas and components from prior diffusion-based point-cloud reconstruction methods. We thank the authors of the related open-source projects and the reviewers for their valuable feedback.

## License

The license for the source code and the terms of use for the YUPU dataset will be provided upon release.

## Contact

For questions regarding this project, please contact:

**Sunghwan (Jacob) Yoo**
York University
[EMAIL_ADDRESS]
