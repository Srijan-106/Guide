# Guide (VisionAssist)

A software-first AI accessibility assistant for blind and visually impaired users.

## Project Status
**In active development —**
- **Day 0** — Research and planning phase completed. No functional code yet.
- **Day 1** — Python virtual environment set up. GPU-accelerated PyTorch (CUDA 12.8) verified working with RTX 5060. No functional application code yet.
- **Day 2** — First working object detection module built and tested on a live webcam feed using YOLOv8n, running on GPU.

## Development Log
See full day-by-day progress in `docs/devlog.md`.

## Problem
Blind and visually impaired individuals often lack affordable, real-time,
privacy-respecting tools to understand their immediate surroundings — including
identifying objects, reading printed text, recognizing currency, and getting
natural spoken descriptions of a scene.

## Planned Core Features
- Real-time object detection (what is around the user)
- Optical Character Recognition, meaning automated text reading from images (reading signs, labels, documents)
- Currency recognition, trained specifically for Indian currency notes
- Local, offline large language model integration for natural spoken scene descriptions

## Why This Project
Built as a first major end-to-end Data Science / Machine Learning / Computer Vision
project, with an emphasis on real evaluation, genuine understanding of every
component, and honest documentation of the development process — including
failures and debugging, not just final results.

## Research Foundation
This project was built following a structured review of 2024-2026 academic
literature and existing open-source/commercial systems in the assistive
technology space, to identify genuine, honest gaps rather than assuming novelty.

## Development Log
See day-by-day progress in `docs/devlog.md`.