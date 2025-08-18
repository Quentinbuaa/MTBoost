# MTBoost: Reinforcement Learning for Robust DNN Classifiers

MTBoost is a reinforcement learning–based algorithm designed to improve the **robustness** of deep neural network (DNN) classifiers, evaluated through **metamorphic relations (MRs)**.  

## 🚀 Introduction  

Traditional metrics like **accuracy** measure how often a DNN predicts the correct label:  

\[
\text{Accuracy} = \Pr\{F(x) = y\}
\]  

However, accuracy alone does not capture **robustness**. Robustness refers to whether a DNN produces consistent predictions under transformations that should not change the label (e.g., image rotation).  

We define robustness using metamorphic relations (MRs), such as:  

\[
F(x) = F(x') \quad \text{where } x' = T(x)
\]  

MTBoost improves robustness **without sacrificing accuracy** by optimizing three objectives:  

1. Maximize \(\Pr\{F(x) = y\}\)  
2. Maximize \(\Pr\{F(x') = y\}\)  
3. Maximize \(\Pr\{f(x) = f(x')\}\)  

where \(F(x)\) is the predicted label and \(f(x)\) is the raw output (logits).  

## ✨ Features  

- Reinforcement learning–based training algorithm  
- Robustness evaluation using metamorphic relations (MRs)  
- Custom loss function with regularization:  

\[
\text{Loss} = \text{loss}_1 + \alpha \cdot \text{loss}_2 + \beta \cdot \text{loss}_3
\]  

- Applicable to image classification tasks and beyond  

## 📦 Installation  

Clone this repository and install dependencies:  

```bash
git clone https://github.com/your-username/MTBoost.git
cd MTBoost
pip install -r requirements.txt
```  

## 🛠 Usage  

Run training with:  

```bash
python train.py --dataset CIFAR10 --epochs 100 --alpha 0.5 --beta 0.3
```  

Evaluate robustness:  

```bash
python evaluate.py --dataset CIFAR10 --checkpoint checkpoints/model.pth
```  

## 📊 Experimental Results  

We evaluated MTBoost on benchmark datasets such as CIFAR-10 and MNIST.  
- Improved robustness (measured via MR satisfaction)  
- Accuracy maintained within 1% of baseline models  

(Detailed results can be added here.)  

## 📖 Citation  

If you use MTBoost in your research, please cite:  

```
@article{yourpaper2025,
  title={MTBoost: Reinforcement Learning for Robust DNN Classifiers},
  author={Your Name},
  year={2025},
  journal={arXiv preprint arXiv:xxxx.xxxxx}
}
```  

## 📜 License  

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  
