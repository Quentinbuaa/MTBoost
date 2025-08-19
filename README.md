# MTBoost: A Reinforcement-Learning-based Training Method for Enhancing the Robustness of Deep Neural Network Classifiers

## 🚀 Introduction
Traditional metrics like **accuracy** measure how often a DNN predicts the correct label:  

`
Accuracy = Pr{F(x) = y}
`

However, accuracy alone does not capture **robustness**. Robustness refers to whether a DNN produces consistent predictions under transformations that should not change the label (e.g., image rotation).  

Robustness can be evaluated with metamorphic relations (MRs), such as:  

``
F(x) = F(x') where  x' = T(x;θ), a transformation with x.
``


MTBoost improves robustness **without sacrificing accuracy** by optimizing three objectives:  

1. Maximize `Pr{F(x) = y}`  
2. Maximize `Pr{F(x') = y}`  
3. Maximize `Pr{f(x) = f(x')}`  

where `F(x)` is the predicted label and `f(x)` is the raw output (logits).  

## ✨ Features  

- Custom loss function with regularization:

``
Loss = loss1 +  α·loss2 + β·loss3, where, if loss1 = cross_entropy(F(x), y), loss2=cross_entropy(F(x'), y), and loss3=KLDivLoss(f(x'), f(x))
``

- Reinforcement learning–based training algorithm is designed to select `x'=T(x;θ)` after each train epoch.

## 📦 Installation  

Clone this repository and install dependencies:  

```bash
git clone https://github.com/your-username/MTBoost.git
cd MTBoost
pip install -r requirements.txt
```  

## 🛠 Usage  

Run all the experiments within our paper:  

```bash
python exp_platform.py 
```  


## 📊 Experimental Results  

We evaluated MTBoost on benchmark datasets such as CIFAR-10, SVHN, FashionMNIST, and GTSRB.  
- Improved robustness (measured via MR satisfaction)  
- Accuracy maintained compared to the baseline models  

(Detailed results can be added here.)  

## 📖 Citation  

If you use MTBoost in your research, please cite:  

```
@article{yourpaper2025,
  title={MT-Boost: A Metamorphic-Testing Based Training Method for Enhancing the Robustness of Deep Neural Network Classifiers},
  author={Kun Qiu, Yu Zhou, Pak-Lok Poon, Tsong-Yueh Chen},
  year={2025},
  journal={Information and Software Technology}
}
```  

## 📜 License  

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.  
