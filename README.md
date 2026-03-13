# NanoMoE-Instruct

**NanoMoE-Instruct** is a lightweight, sparse Mixture-of-Experts (MoE) large language model framework designed specifically for edge computing and vertical domains, such as Finance and Healthcare. 

This project implements an end-to-end training pipeline, covering Pre-training, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Proximal Policy Optimization (PPO), making it a comprehensive platform for building and aligning domain-specific capabilities on low-resource devices.

## 🌟 Key Features

* **Sparse Mixture of Experts (MoE)**: Integrates an MoE architecture with a custom routing mechanism to significantly expand model capacity and handle multi-domain representations without proportionally increasing inference compute (FLOPs).
* **From-Scratch Core Operators**: Independently implemented foundational Transformer components in PyTorch, including:
  * **Grouped Query Attention (GQA)** for optimized memory bandwidth.
  * **Rotary Position Embedding (RoPE)** for efficient positional encoding.
  * **RMSNorm** and SwiGLU activations.
* **End-to-End Alignment Pipeline**: A complete RLHF/Alignment workflow featuring robust scripts for:
  * Supervised Fine-Tuning (SFT) & parameter-efficient LoRA tuning.
  * Direct Preference Optimization (DPO).
  * Proximal Policy Optimization (PPO).
* **Edge-Friendly Design**: Tailored to run efficiently on local edge servers and customized for vertical data domains (handling medical QA and financial/customer-service protocols).

## 📂 Project Structure

* `model/`: Core NanoMoE model implementations and tokenizer configurations.
* `method/`: Standalone implementation of fundamental operators (GQA, MoE, RoPE, RMSNorm).
* `dataset/`: Dataset processing for pre-training, SFT, and alignment (DPO/RLAIF), including domain-specific samples.
* `trainer/`: Comprehensive training loop implementations for all stages of the LLM lifecycle.
* `main.py` / `eval.py`: Entry points for generation, testing, and evaluation.

## 🙏 Acknowledgements

This project was built upon and heavily inspired by the excellent [minimind](https://github.com/jingyaogong/minimind) repository. A huge thanks to the original authors for providing a fantastic structural reference for building minimal language models from scratch!
