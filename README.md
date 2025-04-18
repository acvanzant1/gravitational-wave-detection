**Gravitational Wave Detection Using Deep Learning**

**Author: Adam C. Van Zant**

**Institution: University of Louisville**

**Overview**

This project investigates the use of deep learning for the real-time classification of gravitational wave (GW) signals from binary black hole (BBH) mergers. Two architectures are explored and compared:
	•	A high-capacity 1D Convolutional Neural Network (1D-CNN)
	•	A parameter-efficient Residual Network (ResNet)

Both models are implemented and trained entirely on Apple Silicon (M2 Max) hardware using public strain data from the Gravitational Wave Open Science Center (GWOSC). The goal is to evaluate their performance in real LIGO noise environments while ensuring portability to consumer-grade, non-GPU systems.

**Key Contributions**
	•	Fully local training and inference: No cloud services or NVIDIA GPUs were used.
	•	Model benchmarking on Apple Silicon: Demonstrates strong classification performance with minimal resource overhead.
	•	Open and reproducible pipeline: All data and models are derived from open-source APIs and tools such as GWPy, PyCBC, TensorFlow, and PyTorch.
	•	Perfect classification results: Both models achieved 100% precision, recall, and F1-score on test data under current settings.
	•	Realistic preprocessing: Includes whitening, bandpass filtering, and injection of SEOBNRv2 waveforms into both Gaussian and PSD-shaped synthetic noise.

**Hardware & Software**
	•	Platform: Apple Mac Studio (M2 Max, 32 GB RAM)
	•	Python version: 3.9.9
	•	Frameworks: TensorFlow (Metal), PyTorch (MPS), GWPy, PyCBC
	•	Runtime:
	•	1D-CNN: ~10.5 min
	•	ResNet: ~25 min

**Getting Started**

To reproduce the results, follow the setup instructions in the Experimental Procedure section of the thesis. Briefly:
	1.	Clone this repo.
	2.	Replace local file paths.
	3.	Setup Python 3.9.9 virtual environment.
	4.	Install dependencies using requirements.txt.
	5.	Run scripts in order: data download → preprocessing → training.

**Directory Structure**

 models/                   # Contains both CNN and ResNet implementations  
scripts/                  # Data acquisition and preprocessing notebooks  
requirements.txt          # Dependency list  


**Citations**


This work relies on data from the Gravitational Wave Open Science Center (GWOSC) and uses scientific libraries under open-source licenses. All waveform injections are generated via the SEOBNRv2 approximant using PyCBC.

## 📄 Master’s Thesis

You can view my full thesis document here:  
🔗 [Gravitational Wave Detection Using Deep Learning (View Only)](https://docs.google.com/document/d/12lWyfPhXyMHmYyitk9l8vDl4AnJroP2X9f_7rGCKUAc/edit?usp=sharing)

This paper outlines the development, training, and evaluation of both 1D-CNN and ResNet architectures for classifying gravitational wave signals in realistic LIGO noise, with full compatibility on Apple Silicon hardware.

Note, the thesis followed a structural protocol found on the syllabus - it may contain unconventional subsections, but overall this thesis satisifed the necessary requirments for me to graduate. If you have any questions regarding the project or have any suggestions/improvements, feel free to reach out to my personal email: @adamvanzant40@gmail.com . Thank's for taking the time to check out my work! - Adam
