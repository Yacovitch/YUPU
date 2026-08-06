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
