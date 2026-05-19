"""Rich ML/AI/CS knowledge graph with BFS path finding, prerequisites, explanations, and quizzes."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Concept:
    """A single node in the knowledge graph."""

    name: str
    description: str
    prerequisites: list[str]
    difficulty: str  # beginner / intermediate / advanced
    category: str
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph data — 60+ ML/AI/CS concepts
# ---------------------------------------------------------------------------

_CONCEPTS: dict[str, Concept] = {
    # ---- Foundations --------------------------------------------------------
    "linear_algebra": Concept(
        name="Linear Algebra",
        description=(
            "Linear algebra is the branch of mathematics dealing with vector spaces and linear "
            "mappings between them. It provides the mathematical backbone for nearly all modern "
            "machine learning, including matrix operations, eigendecomposition, and dot products. "
            "Understanding linear algebra is essential for reasoning about neural network weights, "
            "principal component analysis, and data transformations."
        ),
        prerequisites=[],
        difficulty="beginner",
        category="mathematics",
        aliases=["linalg"],
    ),
    "calculus": Concept(
        name="Calculus",
        description=(
            "Calculus is the mathematical study of continuous change, encompassing derivatives and "
            "integrals. In machine learning it underpins gradient-based optimisation: derivatives "
            "measure how a loss function changes with respect to parameters. Without calculus, "
            "backpropagation and most optimisation algorithms cannot be understood or implemented."
        ),
        prerequisites=[],
        difficulty="beginner",
        category="mathematics",
        aliases=["differential calculus"],
    ),
    "probability": Concept(
        name="Probability Theory",
        description=(
            "Probability theory provides a formal framework for reasoning under uncertainty. "
            "It defines events, random variables, distributions, and expectations. Machine "
            "learning models are fundamentally probabilistic: classifiers estimate class "
            "probabilities, generative models learn data distributions, and Bayesian methods "
            "reason over posterior beliefs."
        ),
        prerequisites=[],
        difficulty="beginner",
        category="mathematics",
        aliases=["probability", "statistics"],
    ),
    "statistics": Concept(
        name="Statistics",
        description=(
            "Statistics is the discipline of collecting, analysing, and interpreting data. "
            "It includes descriptive statistics (mean, variance, correlation) and inferential "
            "statistics (hypothesis testing, confidence intervals). Statistical thinking is "
            "required for evaluating model performance, avoiding overfitting, and designing "
            "rigorous experiments."
        ),
        prerequisites=["probability"],
        difficulty="beginner",
        category="mathematics",
        aliases=["statistical inference"],
    ),
    "information_theory": Concept(
        name="Information Theory",
        description=(
            "Information theory quantifies information content using entropy, mutual information, "
            "and KL-divergence. In ML, it explains why cross-entropy is a natural loss function "
            "for classification: minimising cross-entropy is equivalent to maximum likelihood "
            "estimation. It also grounds the concept of model capacity and compression."
        ),
        prerequisites=["probability"],
        difficulty="intermediate",
        category="mathematics",
        aliases=["information theory"],
    ),
    # ---- Core ML ------------------------------------------------------------
    "machine_learning": Concept(
        name="Machine Learning",
        description=(
            "Machine learning is the field of algorithms that learn patterns from data rather than "
            "being explicitly programmed. It encompasses supervised, unsupervised, and reinforcement "
            "learning paradigms. The central challenge is generalisation: learning a model on "
            "training data that performs well on unseen examples."
        ),
        prerequisites=["linear_algebra", "probability", "calculus"],
        difficulty="beginner",
        category="machine_learning",
        aliases=["ML"],
    ),
    "loss_function": Concept(
        name="Loss Function",
        description=(
            "A loss function (cost function) measures how far a model's predictions are from the "
            "ground truth. Common choices include mean squared error for regression and cross-entropy "
            "for classification. The choice of loss function encodes assumptions about the "
            "data-generating process and directly determines what the model optimises."
        ),
        prerequisites=["machine_learning"],
        difficulty="beginner",
        category="machine_learning",
        aliases=["cost function", "objective function"],
    ),
    "cross_entropy": Concept(
        name="Cross-Entropy Loss",
        description=(
            "Cross-entropy measures the divergence between a predicted probability distribution "
            "and the true distribution. For binary classification, it equals "
            "-y*log(p) - (1-y)*log(1-p). Minimising cross-entropy is equivalent to maximising "
            "log-likelihood, making it the canonical loss for probabilistic classifiers. "
            "It is derived directly from information theory."
        ),
        prerequisites=["loss_function", "information_theory"],
        difficulty="intermediate",
        category="machine_learning",
        aliases=["log loss", "categorical cross-entropy"],
    ),
    "gradient_descent": Concept(
        name="Gradient Descent",
        description=(
            "Gradient descent is an iterative optimisation algorithm that updates parameters in "
            "the direction of steepest descent of the loss landscape. Each step subtracts the "
            "gradient scaled by a learning rate. Variants include stochastic gradient descent "
            "(SGD), mini-batch SGD, and momentum-based methods like Adam."
        ),
        prerequisites=["calculus", "loss_function"],
        difficulty="beginner",
        category="optimisation",
        aliases=["SGD", "gradient optimisation"],
    ),
    "backpropagation": Concept(
        name="Backpropagation",
        description=(
            "Backpropagation is the algorithm for computing gradients of a loss with respect to "
            "every parameter in a neural network by applying the chain rule of calculus layer by "
            "layer from output back to input. It makes training deep networks computationally "
            "feasible. Automatic differentiation frameworks implement backprop under the hood."
        ),
        prerequisites=["gradient_descent", "calculus", "neural_network"],
        difficulty="intermediate",
        category="optimisation",
        aliases=["backprop", "reverse-mode autodiff"],
    ),
    "regularisation": Concept(
        name="Regularisation",
        description=(
            "Regularisation techniques reduce overfitting by adding a penalty on model complexity "
            "to the loss function. L2 (weight decay) penalises large weights; L1 (lasso) promotes "
            "sparsity; dropout randomly zeroes activations during training. Regularisation "
            "encodes a prior that simpler models generalise better."
        ),
        prerequisites=["loss_function", "overfitting"],
        difficulty="intermediate",
        category="machine_learning",
        aliases=["L1", "L2", "weight decay", "dropout"],
    ),
    "overfitting": Concept(
        name="Overfitting",
        description=(
            "Overfitting occurs when a model memorises training data instead of learning the "
            "underlying distribution, resulting in low training loss but high test loss. It is "
            "diagnosed by a large train-test performance gap. Prevention strategies include "
            "regularisation, early stopping, data augmentation, and choosing simpler model "
            "architectures."
        ),
        prerequisites=["machine_learning", "statistics"],
        difficulty="beginner",
        category="machine_learning",
        aliases=["generalisation gap"],
    ),
    # ---- Neural Networks ----------------------------------------------------
    "neural_network": Concept(
        name="Neural Network",
        description=(
            "A neural network is a function composed of layers of parameterised affine "
            "transformations followed by nonlinear activation functions. Inspired loosely by "
            "biological neurons, they are universal function approximators. Depth allows "
            "hierarchical feature learning, and width provides representational capacity."
        ),
        prerequisites=["machine_learning", "linear_algebra"],
        difficulty="beginner",
        category="deep_learning",
        aliases=["deep neural network", "DNN", "multilayer perceptron", "MLP"],
    ),
    "activation_function": Concept(
        name="Activation Function",
        description=(
            "Activation functions introduce nonlinearity into neural networks, allowing them to "
            "represent complex mappings. Without them, stacked linear layers collapse to a single "
            "linear transformation. The choice of activation affects gradient flow, training "
            "stability, and representational power."
        ),
        prerequisites=["neural_network"],
        difficulty="beginner",
        category="deep_learning",
        aliases=["nonlinearity"],
    ),
    "relu": Concept(
        name="ReLU",
        description=(
            "ReLU (Rectified Linear Unit) is defined as f(x) = max(0, x). It avoids the vanishing "
            "gradient problem that plagued sigmoid/tanh networks because its gradient is exactly 1 "
            "for positive inputs. ReLU networks are sparse and computationally cheap. Variants "
            "include Leaky ReLU, ELU, and GELU used in transformers."
        ),
        prerequisites=["activation_function"],
        difficulty="beginner",
        category="deep_learning",
        aliases=["rectified linear unit", "leaky relu"],
    ),
    "sigmoid": Concept(
        name="Sigmoid Function",
        description=(
            "The sigmoid maps real numbers to (0, 1) via sigma(x) = 1/(1+e^-x), making it natural "
            "for binary classification outputs. However, it saturates for large |x| causing "
            "vanishing gradients in deep networks. It is still widely used in output layers "
            "and gating mechanisms such as LSTM gates."
        ),
        prerequisites=["activation_function"],
        difficulty="beginner",
        category="deep_learning",
        aliases=["logistic function"],
    ),
    "tanh": Concept(
        name="Tanh",
        description=(
            "Tanh (hyperbolic tangent) maps inputs to (-1, 1), is zero-centred unlike sigmoid, "
            "and was the dominant hidden-layer activation before ReLU. It still suffers from "
            "vanishing gradients in very deep networks. Tanh is the activation used in LSTM "
            "cell states and some recurrent architectures."
        ),
        prerequisites=["activation_function"],
        difficulty="beginner",
        category="deep_learning",
        aliases=["hyperbolic tangent"],
    ),
    "batch_normalisation": Concept(
        name="Batch Normalisation",
        description=(
            "Batch normalisation normalises layer inputs to zero mean and unit variance within "
            "each mini-batch, then applies learned scale and shift parameters. It stabilises "
            "training, allows higher learning rates, and acts as a mild regulariser. It is a "
            "key building block in ResNets and many modern architectures."
        ),
        prerequisites=["neural_network", "gradient_descent"],
        difficulty="intermediate",
        category="deep_learning",
        aliases=["batch norm", "BN"],
    ),
    # ---- CNNs ---------------------------------------------------------------
    "cnn": Concept(
        name="Convolutional Neural Network",
        description=(
            "CNNs exploit the spatial structure of images through weight sharing: the same filter "
            "is applied across all spatial positions. This dramatically reduces parameters compared "
            "to fully connected networks on images. CNNs build hierarchical features: edges in "
            "early layers, textures in middle layers, semantic parts in deep layers."
        ),
        prerequisites=["neural_network"],
        difficulty="intermediate",
        category="computer_vision",
        aliases=["CNN", "convnet"],
    ),
    "convolution": Concept(
        name="Convolution Operation",
        description=(
            "Convolution slides a learnable kernel over an input feature map, computing dot "
            "products at each position. The kernel size, stride, and padding control the "
            "spatial resolution of the output. Convolutional layers are equivariant to "
            "translation, which is a strong inductive bias for images."
        ),
        prerequisites=["cnn", "linear_algebra"],
        difficulty="intermediate",
        category="computer_vision",
        aliases=["conv layer", "kernel", "filter"],
    ),
    "pooling": Concept(
        name="Pooling Layer",
        description=(
            "Pooling layers downsample feature maps by summarising local regions, providing "
            "spatial invariance and reducing computation. Max pooling selects the largest "
            "activation; average pooling computes the mean. Global average pooling replaces "
            "fully connected heads in modern architectures like MobileNet."
        ),
        prerequisites=["convolution"],
        difficulty="beginner",
        category="computer_vision",
        aliases=["max pooling", "average pooling"],
    ),
    "feature_maps": Concept(
        name="Feature Maps",
        description=(
            "Feature maps are the outputs of convolutional layers, representing different "
            "learned features at each spatial position. Earlier feature maps detect low-level "
            "features like edges; deeper maps encode semantic content. Visualising feature "
            "maps is a primary tool for interpreting CNN behaviour."
        ),
        prerequisites=["convolution"],
        difficulty="intermediate",
        category="computer_vision",
        aliases=["activation maps", "feature volumes"],
    ),
    "residual_network": Concept(
        name="Residual Network (ResNet)",
        description=(
            "ResNets introduce skip connections that add the input of a block directly to its "
            "output. This enables training extremely deep networks (100+ layers) by providing "
            "gradient highways that bypass non-linear transformations. ResNets won ImageNet 2015 "
            "and remain a standard backbone in computer vision."
        ),
        prerequisites=["cnn", "batch_normalisation"],
        difficulty="advanced",
        category="computer_vision",
        aliases=["ResNet", "skip connections", "residual connections"],
    ),
    # ---- Transformers -------------------------------------------------------
    "transformer": Concept(
        name="Transformer",
        description=(
            "The Transformer architecture, introduced in 'Attention Is All You Need' (2017), "
            "processes sequences using self-attention rather than recurrence or convolution. "
            "It consists of encoder and/or decoder stacks with multi-head attention and "
            "feed-forward sublayers. Transformers power all large language models (GPT, BERT, "
            "T5) and have expanded to vision and multimodal domains."
        ),
        prerequisites=["neural_network", "attention", "positional_encoding"],
        difficulty="advanced",
        category="nlp",
        aliases=["transformer model"],
    ),
    "attention": Concept(
        name="Attention Mechanism",
        description=(
            "Attention allows a model to dynamically weight parts of an input when producing "
            "each output element. Given queries, keys, and values, attention scores are computed "
            "as softmax(QK^T / sqrt(d_k))V. It was first introduced for encoder-decoder "
            "sequence-to-sequence models and dramatically improved machine translation quality."
        ),
        prerequisites=["neural_network"],
        difficulty="intermediate",
        category="nlp",
        aliases=["attention layer", "scaled dot-product attention"],
    ),
    "self_attention": Concept(
        name="Self-Attention",
        description=(
            "Self-attention applies the attention mechanism where queries, keys, and values all "
            "come from the same sequence, allowing each position to attend to all others. This "
            "captures long-range dependencies without recurrence, making it the core operation "
            "in Transformer encoders (BERT) and decoders (GPT)."
        ),
        prerequisites=["attention"],
        difficulty="intermediate",
        category="nlp",
        aliases=["intra-attention"],
    ),
    "multi_head_attention": Concept(
        name="Multi-Head Attention",
        description=(
            "Multi-head attention runs several self-attention operations in parallel with "
            "different learned projections (heads), allowing the model to attend to information "
            "from different representation subspaces simultaneously. The outputs are concatenated "
            "and linearly projected. Typical transformers use 8-32 heads."
        ),
        prerequisites=["self_attention"],
        difficulty="advanced",
        category="nlp",
        aliases=["MHA"],
    ),
    "positional_encoding": Concept(
        name="Positional Encoding",
        description=(
            "Since self-attention is permutation-invariant, positional encodings inject "
            "sequence order information into token embeddings. The original Transformer used "
            "fixed sinusoidal encodings; modern models use learned absolute or relative "
            "position embeddings (RoPE, ALiBi). Without them, transformers treat sequences "
            "as unordered sets."
        ),
        prerequisites=["transformer"],
        difficulty="intermediate",
        category="nlp",
        aliases=["position embeddings", "RoPE", "ALiBi"],
    ),
    "embedding": Concept(
        name="Embedding",
        description=(
            "Embeddings are dense vector representations of discrete objects (words, tokens, "
            "items) learned from data. Word2Vec and GloVe are classical word embeddings; "
            "transformer models learn contextual embeddings. Embeddings map high-dimensional "
            "categorical spaces into compact, semantically meaningful vector spaces."
        ),
        prerequisites=["neural_network", "linear_algebra"],
        difficulty="beginner",
        category="nlp",
        aliases=["word embedding", "token embedding"],
    ),
    "language_model": Concept(
        name="Language Model",
        description=(
            "A language model assigns probabilities to sequences of tokens, learning the "
            "statistical structure of text. Autoregressive models (GPT) predict the next token; "
            "masked models (BERT) predict masked tokens. Large language models (LLMs) trained "
            "on web-scale data exhibit emergent capabilities including reasoning and few-shot "
            "learning."
        ),
        prerequisites=["transformer", "embedding"],
        difficulty="advanced",
        category="nlp",
        aliases=["LLM", "GPT", "BERT", "large language model"],
    ),
    # ---- Reinforcement Learning ---------------------------------------------
    "reinforcement_learning": Concept(
        name="Reinforcement Learning",
        description=(
            "Reinforcement learning studies agents that learn optimal behaviour through trial and "
            "error by interacting with an environment and receiving reward signals. It frames "
            "learning as a Markov decision process (MDP). RL has achieved superhuman performance "
            "in games (AlphaGo, Atari) and is used for fine-tuning language models (RLHF)."
        ),
        prerequisites=["machine_learning", "probability"],
        difficulty="advanced",
        category="reinforcement_learning",
        aliases=["RL"],
    ),
    "policy": Concept(
        name="Policy",
        description=(
            "A policy maps observations (states) to actions, defining how an agent behaves. "
            "A deterministic policy selects one action; a stochastic policy outputs a "
            "probability distribution over actions. Policy gradient methods directly optimise "
            "the policy parameters using gradient ascent on expected reward."
        ),
        prerequisites=["reinforcement_learning"],
        difficulty="intermediate",
        category="reinforcement_learning",
        aliases=["policy function", "pi"],
    ),
    "reward": Concept(
        name="Reward Signal",
        description=(
            "The reward signal provides scalar feedback to an RL agent indicating the desirability "
            "of a state-action pair. The agent maximises cumulative discounted reward. Reward "
            "design (reward shaping) is critical and notoriously difficult; poorly designed "
            "rewards lead to reward hacking and unintended agent behaviour."
        ),
        prerequisites=["reinforcement_learning"],
        difficulty="beginner",
        category="reinforcement_learning",
        aliases=["reward function", "return"],
    ),
    "q_learning": Concept(
        name="Q-Learning",
        description=(
            "Q-learning is a model-free RL algorithm that learns the action-value function Q(s,a), "
            "the expected cumulative reward of taking action a in state s. It uses a temporal "
            "difference update rule and is provably convergent in tabular settings. Deep Q-Networks "
            "(DQN) use neural networks to approximate Q for large state spaces."
        ),
        prerequisites=["policy", "reward"],
        difficulty="advanced",
        category="reinforcement_learning",
        aliases=["DQN", "Q-network"],
    ),
    "markov_decision_process": Concept(
        name="Markov Decision Process",
        description=(
            "An MDP is a mathematical framework for sequential decision making defined by a "
            "tuple (S, A, P, R, gamma): states, actions, transition probabilities, rewards, and "
            "a discount factor. The Markov property means the next state depends only on the "
            "current state and action. MDPs are the formal foundation for all of RL."
        ),
        prerequisites=["reinforcement_learning", "probability"],
        difficulty="intermediate",
        category="reinforcement_learning",
        aliases=["MDP"],
    ),
    # ---- Decision Trees and Ensembles ----------------------------------------
    "decision_tree": Concept(
        name="Decision Tree",
        description=(
            "A decision tree partitions the feature space into axis-aligned rectangles by "
            "recursively splitting on features that maximise information gain or reduce Gini "
            "impurity. They are interpretable and handle mixed feature types naturally. However, "
            "individual deep trees overfit badly and have high variance."
        ),
        prerequisites=["machine_learning", "information_theory"],
        difficulty="beginner",
        category="ensemble_methods",
        aliases=["CART", "classification tree", "regression tree"],
    ),
    "random_forest": Concept(
        name="Random Forest",
        description=(
            "Random Forest is a bagging ensemble that trains many decision trees on bootstrap "
            "samples of the data, with each split restricted to a random feature subset. "
            "Aggregating diverse trees drastically reduces variance without increasing bias. "
            "Random forests are robust, require little tuning, and provide feature importances."
        ),
        prerequisites=["decision_tree"],
        difficulty="intermediate",
        category="ensemble_methods",
        aliases=["RF"],
    ),
    "gradient_boosting": Concept(
        name="Gradient Boosting",
        description=(
            "Gradient boosting builds an ensemble additively: each new tree is trained to "
            "predict the residual errors (pseudo-gradients of the loss) of the current ensemble. "
            "It converts weak learners into a powerful predictor through sequential refinement. "
            "It is the dominant algorithm on tabular data competitions."
        ),
        prerequisites=["decision_tree", "gradient_descent"],
        difficulty="advanced",
        category="ensemble_methods",
        aliases=["GBM", "boosting"],
    ),
    "xgboost": Concept(
        name="XGBoost",
        description=(
            "XGBoost is an optimised, regularised implementation of gradient boosting with "
            "second-order Taylor expansion of the loss, built-in L1/L2 regularisation, "
            "column subsampling, and hardware-aware parallelism. It dominated Kaggle "
            "competitions from 2014 to 2017 and remains a go-to baseline for tabular data."
        ),
        prerequisites=["gradient_boosting"],
        difficulty="advanced",
        category="ensemble_methods",
        aliases=["XGB", "extreme gradient boosting"],
    ),
    # ---- Clustering / Unsupervised -------------------------------------------
    "clustering": Concept(
        name="Clustering",
        description=(
            "Clustering discovers groups of similar data points without using labels. Algorithms "
            "include k-means (centroid-based), DBSCAN (density-based), and hierarchical "
            "agglomerative clustering. Clustering is used for customer segmentation, anomaly "
            "detection, and as a pre-processing step for supervised learning."
        ),
        prerequisites=["machine_learning"],
        difficulty="beginner",
        category="unsupervised",
        aliases=["k-means", "DBSCAN"],
    ),
    "dimensionality_reduction": Concept(
        name="Dimensionality Reduction",
        description=(
            "Dimensionality reduction projects high-dimensional data to fewer dimensions while "
            "preserving important structure. PCA finds orthogonal directions of maximum variance; "
            "t-SNE and UMAP preserve local neighbourhoods for visualisation. Reducing dimensions "
            "alleviates the curse of dimensionality and speeds up downstream learning."
        ),
        prerequisites=["linear_algebra", "machine_learning"],
        difficulty="intermediate",
        category="unsupervised",
        aliases=["PCA", "t-SNE", "UMAP"],
    ),
    "autoencoder": Concept(
        name="Autoencoder",
        description=(
            "An autoencoder is a neural network trained to reconstruct its input through a "
            "bottleneck latent representation. The encoder compresses data; the decoder "
            "reconstructs it. Variational autoencoders (VAEs) learn a continuous latent "
            "distribution, enabling generative sampling. Autoencoders are used for "
            "anomaly detection, denoising, and representation learning."
        ),
        prerequisites=["neural_network", "dimensionality_reduction"],
        difficulty="intermediate",
        category="generative_models",
        aliases=["VAE", "variational autoencoder"],
    ),
    # ---- Generative Models ---------------------------------------------------
    "generative_adversarial_network": Concept(
        name="Generative Adversarial Network",
        description=(
            "A GAN consists of a generator that creates synthetic samples and a discriminator "
            "that classifies real vs. fake. They are trained adversarially in a minimax game. "
            "GANs produce highly realistic images (StyleGAN) but are notoriously difficult to "
            "train due to mode collapse and unstable dynamics."
        ),
        prerequisites=["neural_network", "probability"],
        difficulty="advanced",
        category="generative_models",
        aliases=["GAN"],
    ),
    "diffusion_model": Concept(
        name="Diffusion Model",
        description=(
            "Diffusion models learn to reverse a gradual noising process: they are trained to "
            "predict and remove noise from corrupted data at each step. During generation, pure "
            "noise is iteratively denoised. Diffusion models (DALL-E 3, Stable Diffusion) "
            "surpassed GANs in image generation quality and training stability."
        ),
        prerequisites=["neural_network", "probability", "score_function"],
        difficulty="advanced",
        category="generative_models",
        aliases=["DDPM", "score-based model"],
    ),
    "score_function": Concept(
        name="Score Function",
        description=(
            "The score function is the gradient of the log probability density ∇_x log p(x). "
            "Score matching trains a neural network to approximate this gradient, enabling "
            "sampling from complex distributions without computing the partition function. "
            "It is the theoretical foundation for score-based generative models and diffusion."
        ),
        prerequisites=["probability", "calculus"],
        difficulty="advanced",
        category="generative_models",
        aliases=["score matching"],
    ),
    # ---- Recurrent Architectures --------------------------------------------
    "rnn": Concept(
        name="Recurrent Neural Network",
        description=(
            "RNNs process sequences by maintaining a hidden state updated at each time step. "
            "They share parameters across time steps, making them suitable for variable-length "
            "sequences. Vanilla RNNs suffer from vanishing/exploding gradients, making it "
            "hard to learn long-range dependencies."
        ),
        prerequisites=["neural_network"],
        difficulty="intermediate",
        category="nlp",
        aliases=["RNN", "recurrent network"],
    ),
    "lstm": Concept(
        name="Long Short-Term Memory",
        description=(
            "LSTM cells use gating mechanisms (input, forget, output gates) to control "
            "information flow through time, alleviating the vanishing gradient problem. "
            "They maintain a separate cell state that can carry information across many "
            "time steps. LSTMs powered the state of the art in NLP before Transformers."
        ),
        prerequisites=["rnn", "sigmoid", "tanh"],
        difficulty="advanced",
        category="nlp",
        aliases=["LSTM"],
    ),
    # ---- Evaluation ---------------------------------------------------------
    "cross_validation": Concept(
        name="Cross-Validation",
        description=(
            "Cross-validation estimates model generalisation by partitioning data into k folds "
            "and training/evaluating k times, each time using a different fold as the test set. "
            "k-fold CV gives a lower-variance estimate of test performance than a single "
            "train/test split, especially with limited data."
        ),
        prerequisites=["overfitting", "statistics"],
        difficulty="beginner",
        category="evaluation",
        aliases=["k-fold CV", "k-fold cross-validation"],
    ),
    "roc_auc": Concept(
        name="ROC-AUC",
        description=(
            "The Receiver Operating Characteristic curve plots true positive rate vs. false "
            "positive rate at varying classification thresholds. The Area Under the Curve (AUC) "
            "summarises the entire curve as a single number: 0.5 is random, 1.0 is perfect. "
            "AUC is threshold-independent, making it a robust metric for class-imbalanced data."
        ),
        prerequisites=["statistics", "machine_learning"],
        difficulty="intermediate",
        category="evaluation",
        aliases=["AUC", "AUROC"],
    ),
    "precision_recall": Concept(
        name="Precision and Recall",
        description=(
            "Precision is the fraction of positive predictions that are correct; recall is the "
            "fraction of actual positives that are detected. The F1-score is their harmonic mean. "
            "Precision-recall tradeoffs are central in information retrieval, medical diagnostics, "
            "and any domain where one error type is much costlier than the other."
        ),
        prerequisites=["machine_learning"],
        difficulty="beginner",
        category="evaluation",
        aliases=["F1 score", "precision", "recall"],
    ),
    # ---- Optimisation -------------------------------------------------------
    "adam_optimizer": Concept(
        name="Adam Optimizer",
        description=(
            "Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning rates "
            "using first-moment (mean) and second-moment (uncentred variance) estimates of "
            "gradients. It combines momentum and RMSProp. Adam is the default optimiser for "
            "training transformers and most deep learning models."
        ),
        prerequisites=["gradient_descent"],
        difficulty="intermediate",
        category="optimisation",
        aliases=["Adam", "AdamW"],
    ),
    "learning_rate_schedule": Concept(
        name="Learning Rate Schedule",
        description=(
            "A learning rate schedule adjusts the step size during training. Common schedules "
            "include step decay, cosine annealing, and linear warmup followed by decay. "
            "Warmup is especially important for transformers: starting with a small learning "
            "rate prevents early training instability. The schedule is often as important as "
            "the optimizer choice."
        ),
        prerequisites=["gradient_descent"],
        difficulty="intermediate",
        category="optimisation",
        aliases=["LR schedule", "cosine annealing", "warmup"],
    ),
    # ---- Transfer Learning --------------------------------------------------
    "transfer_learning": Concept(
        name="Transfer Learning",
        description=(
            "Transfer learning reuses representations learned on a source task for a different "
            "target task. A pre-trained model is fine-tuned on task-specific data, drastically "
            "reducing the data and compute required. Pre-training on large corpora followed by "
            "supervised fine-tuning is the dominant paradigm in NLP and computer vision."
        ),
        prerequisites=["neural_network", "machine_learning"],
        difficulty="intermediate",
        category="deep_learning",
        aliases=["fine-tuning", "domain adaptation"],
    ),
    "fine_tuning": Concept(
        name="Fine-Tuning",
        description=(
            "Fine-tuning adapts a pre-trained model to a specific task by continuing training "
            "on task-specific data with a small learning rate. Full fine-tuning updates all "
            "weights; parameter-efficient methods (LoRA, adapters, prompt tuning) update only "
            "a small fraction of parameters, making fine-tuning feasible on consumer hardware."
        ),
        prerequisites=["transfer_learning"],
        difficulty="advanced",
        category="deep_learning",
        aliases=["LoRA", "PEFT", "adapter tuning"],
    ),
    # ---- RAG / Retrieval ----------------------------------------------------
    "retrieval_augmented_generation": Concept(
        name="Retrieval-Augmented Generation",
        description=(
            "RAG augments language model generation by first retrieving relevant documents from "
            "an external knowledge base, then conditioning generation on both the query and "
            "retrieved context. It reduces hallucination and allows knowledge to be updated "
            "without retraining the model. RAG is the dominant architecture for enterprise "
            "LLM applications."
        ),
        prerequisites=["language_model", "embedding"],
        difficulty="advanced",
        category="nlp",
        aliases=["RAG"],
    ),
    "vector_database": Concept(
        name="Vector Database",
        description=(
            "A vector database stores dense embedding vectors and supports efficient "
            "approximate nearest-neighbour (ANN) search. Engines like FAISS, Pinecone, "
            "Weaviate, and Chroma enable semantic retrieval at scale. Vector databases are "
            "the storage backbone for RAG systems and recommendation engines."
        ),
        prerequisites=["embedding", "dimensionality_reduction"],
        difficulty="intermediate",
        category="infrastructure",
        aliases=["vector store", "FAISS", "Pinecone"],
    ),
    # ---- Agents / RLHF ------------------------------------------------------
    "rlhf": Concept(
        name="Reinforcement Learning from Human Feedback",
        description=(
            "RLHF fine-tunes language models to align with human preferences. A reward model "
            "is trained on human comparisons between model outputs, then the LLM is optimised "
            "with PPO to maximise this reward. RLHF powered InstructGPT and ChatGPT, demonstrating "
            "that alignment can substantially improve helpfulness and reduce harmful outputs."
        ),
        prerequisites=["language_model", "reinforcement_learning"],
        difficulty="advanced",
        category="alignment",
        aliases=["RLHF", "InstructGPT"],
    ),
    "chain_of_thought": Concept(
        name="Chain-of-Thought Prompting",
        description=(
            "Chain-of-thought (CoT) prompting elicits step-by-step reasoning from language "
            "models by including reasoning traces in few-shot examples or via 'Let's think step "
            "by step'. CoT dramatically improves performance on arithmetic, commonsense, and "
            "symbolic reasoning tasks. It is a key prompting technique for deploying LLMs."
        ),
        prerequisites=["language_model"],
        difficulty="intermediate",
        category="prompting",
        aliases=["CoT", "chain of thought"],
    ),
    "neural_architecture_search": Concept(
        name="Neural Architecture Search",
        description=(
            "NAS automates the design of neural network architectures by framing it as a "
            "search problem over architecture space. Techniques include reinforcement learning "
            "over architecture decisions, evolutionary algorithms, and differentiable NAS "
            "(DARTS). NAS discovered EfficientNet, which dominated ImageNet accuracy-efficiency "
            "tradeoffs."
        ),
        prerequisites=["neural_network", "reinforcement_learning"],
        difficulty="advanced",
        category="meta_learning",
        aliases=["NAS", "AutoML"],
    ),
    "knowledge_distillation": Concept(
        name="Knowledge Distillation",
        description=(
            "Knowledge distillation trains a smaller student model to mimic the soft output "
            "distributions (logits) of a larger teacher model, rather than only hard labels. "
            "The student learns richer information about class similarities. Distillation "
            "produces compact models with near-teacher accuracy, enabling deployment on "
            "resource-constrained devices."
        ),
        prerequisites=["neural_network", "transfer_learning"],
        difficulty="advanced",
        category="efficiency",
        aliases=["model distillation"],
    ),
    "quantisation": Concept(
        name="Quantisation",
        description=(
            "Quantisation reduces model size and inference latency by representing weights "
            "and/or activations with fewer bits (e.g., INT8 or INT4 instead of FP32). "
            "Post-training quantisation requires no retraining; quantisation-aware training "
            "fine-tunes with simulated quantisation. GPTQ and AWQ quantise LLMs to run on "
            "consumer GPUs."
        ),
        prerequisites=["neural_network"],
        difficulty="advanced",
        category="efficiency",
        aliases=["INT8", "INT4", "GPTQ"],
    ),
}

# ---------------------------------------------------------------------------
# Adjacency list: directed edges (prerequisite → dependent)
# ---------------------------------------------------------------------------

_EDGES: list[tuple[str, str]] = [
    # Math → ML
    ("linear_algebra", "machine_learning"),
    ("calculus", "machine_learning"),
    ("probability", "machine_learning"),
    ("probability", "statistics"),
    ("probability", "information_theory"),
    ("calculus", "gradient_descent"),
    ("calculus", "score_function"),
    # ML → Loss
    ("machine_learning", "loss_function"),
    ("machine_learning", "overfitting"),
    ("machine_learning", "clustering"),
    ("machine_learning", "reinforcement_learning"),
    ("machine_learning", "decision_tree"),
    ("machine_learning", "transfer_learning"),
    ("machine_learning", "precision_recall"),
    ("machine_learning", "roc_auc"),
    # Loss chain
    ("loss_function", "gradient_descent"),
    ("loss_function", "cross_entropy"),
    ("loss_function", "regularisation"),
    ("information_theory", "cross_entropy"),
    ("information_theory", "decision_tree"),
    # Optimisation
    ("gradient_descent", "backpropagation"),
    ("gradient_descent", "adam_optimizer"),
    ("gradient_descent", "learning_rate_schedule"),
    ("gradient_descent", "gradient_boosting"),
    # Neural Networks
    ("machine_learning", "neural_network"),
    ("linear_algebra", "neural_network"),
    ("neural_network", "activation_function"),
    ("neural_network", "cnn"),
    ("neural_network", "rnn"),
    ("neural_network", "autoencoder"),
    ("neural_network", "generative_adversarial_network"),
    ("neural_network", "embedding"),
    ("neural_network", "batch_normalisation"),
    ("neural_network", "transfer_learning"),
    ("neural_network", "quantisation"),
    ("neural_network", "knowledge_distillation"),
    ("neural_network", "neural_architecture_search"),
    ("backpropagation", "neural_network"),  # mutual dependency resolved in graph
    # Activations
    ("activation_function", "relu"),
    ("activation_function", "sigmoid"),
    ("activation_function", "tanh"),
    # CNN
    ("cnn", "convolution"),
    ("cnn", "residual_network"),
    ("convolution", "pooling"),
    ("convolution", "feature_maps"),
    ("batch_normalisation", "residual_network"),
    # RNN/LSTM
    ("rnn", "lstm"),
    ("sigmoid", "lstm"),
    ("tanh", "lstm"),
    # Transformer
    ("neural_network", "attention"),
    ("attention", "self_attention"),
    ("self_attention", "multi_head_attention"),
    ("multi_head_attention", "transformer"),
    ("positional_encoding", "transformer"),
    ("transformer", "language_model"),
    ("embedding", "language_model"),
    ("transformer", "positional_encoding"),
    # Unsupervised
    ("linear_algebra", "dimensionality_reduction"),
    ("machine_learning", "dimensionality_reduction"),
    ("dimensionality_reduction", "autoencoder"),
    ("probability", "generative_adversarial_network"),
    ("probability", "diffusion_model"),
    ("score_function", "diffusion_model"),
    # RL
    ("reinforcement_learning", "policy"),
    ("reinforcement_learning", "reward"),
    ("reinforcement_learning", "markov_decision_process"),
    ("probability", "markov_decision_process"),
    ("policy", "q_learning"),
    ("reward", "q_learning"),
    # Ensembles
    ("decision_tree", "random_forest"),
    ("decision_tree", "gradient_boosting"),
    ("gradient_boosting", "xgboost"),
    # Evaluation
    ("statistics", "cross_validation"),
    ("overfitting", "cross_validation"),
    ("statistics", "roc_auc"),
    # Embedding / RAG
    ("embedding", "vector_database"),
    ("dimensionality_reduction", "vector_database"),
    ("language_model", "retrieval_augmented_generation"),
    ("embedding", "retrieval_augmented_generation"),
    # Alignment / RLHF
    ("language_model", "rlhf"),
    ("reinforcement_learning", "rlhf"),
    ("language_model", "chain_of_thought"),
    # Transfer / Distillation
    ("transfer_learning", "fine_tuning"),
    ("transfer_learning", "knowledge_distillation"),
    ("neural_network", "neural_architecture_search"),
    ("reinforcement_learning", "neural_architecture_search"),
    # Regularisation
    ("overfitting", "regularisation"),
]


class KnowledgeGraph:
    """Rich ML/AI/CS knowledge graph with path-finding and explanation generation."""

    def __init__(self) -> None:
        self._concepts: dict[str, Concept] = _CONCEPTS
        # Build adjacency: key → set of neighbours (successors)
        self._adj: dict[str, set[str]] = {k: set() for k in self._concepts}
        # Build reverse adjacency: key → set of predecessors
        self._radj: dict[str, set[str]] = {k: set() for k in self._concepts}
        for src, dst in _EDGES:
            if src in self._adj and dst in self._adj:
                self._adj[src].add(dst)
                self._radj[dst].add(src)
        # Build alias lookup
        self._alias_map: dict[str, str] = {}
        for key, concept in self._concepts.items():
            self._alias_map[key.lower()] = key
            self._alias_map[concept.name.lower()] = key
            for alias in concept.aliases:
                self._alias_map[alias.lower()] = key

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _resolve(self, topic: str) -> str | None:
        """Return canonical key for a topic string, or None if not found."""
        cleaned = topic.strip().lower()
        if cleaned in self._alias_map:
            return self._alias_map[cleaned]
        # Fuzzy: check if topic is a substring of any alias
        for alias, key in self._alias_map.items():
            if cleaned in alias or alias in cleaned:
                return key
        return None

    def _require(self, topic: str) -> str:
        key = self._resolve(topic)
        if key is None:
            raise KeyError(f"Unknown concept: '{topic}'. Call list_concepts() for valid names.")
        return key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_concepts(self) -> list[dict]:
        """Return all concepts with their metadata."""
        return [
            {
                "id": key,
                "name": c.name,
                "category": c.category,
                "difficulty": c.difficulty,
                "description_preview": c.description[:120] + "...",
            }
            for key, c in self._concepts.items()
        ]

    def find_path(self, source: str, target: str) -> list[str]:
        """BFS shortest path from source to target (returns list of concept keys)."""
        src = self._require(source)
        tgt = self._require(target)
        if src == tgt:
            return [src]
        # BFS
        visited: set[str] = {src}
        queue: deque[list[str]] = deque([[src]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            for neighbour in self._adj.get(node, set()):
                if neighbour == tgt:
                    return [*path, neighbour]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append([*path, neighbour])
        return []  # No path found

    def get_prerequisites(self, topic: str) -> list[str]:
        """Return all transitive prerequisites for topic (topological order)."""
        key = self._require(topic)
        visited: set[str] = set()
        result: list[str] = []

        def _dfs(node: str) -> None:
            for parent in self._radj.get(node, set()):
                if parent not in visited:
                    visited.add(parent)
                    _dfs(parent)
                    result.append(parent)

        _dfs(key)
        return result

    def get_related(self, topic: str, depth: int = 1) -> list[str]:
        """Return concepts within `depth` hops (predecessors + successors)."""
        key = self._require(topic)
        visited: set[str] = {key}
        frontier: set[str] = {key}
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self._adj.get(node, set()))
                next_frontier.update(self._radj.get(node, set()))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
        visited.remove(key)
        return sorted(visited)

    def explain(self, topic: str) -> str:
        """Generate a multi-paragraph explanation using graph context."""
        key = self._require(topic)
        concept = self._concepts[key]
        prereqs = self.get_prerequisites(topic)
        successors = sorted(self._adj.get(key, set()))

        parts: list[str] = []

        # Opening paragraph: the concept itself
        parts.append(
            f"## {concept.name}\n\n"
            f"**Category:** {concept.category.replace('_', ' ').title()}  "
            f"**Difficulty:** {concept.difficulty.capitalize()}\n\n"
            f"{concept.description}"
        )

        # Prerequisites section
        if prereqs:
            prereq_names = [self._concepts[p].name for p in prereqs if p in self._concepts]
            parts.append(
                f"### Prerequisites\n\n"
                f"To understand {concept.name} you should first be comfortable with: "
                + ", ".join(f"**{n}**" for n in prereq_names)
                + "."
            )

        # What you can learn next
        if successors:
            succ_names = [self._concepts[s].name for s in successors if s in self._concepts]
            parts.append(
                f"### What {concept.name} Enables\n\n"
                f"Mastering {concept.name} will unlock your understanding of: "
                + ", ".join(f"**{n}**" for n in succ_names[:6])
                + ("..." if len(succ_names) > 6 else ".")
            )

        return "\n\n".join(parts)

    def learning_path(self, target: str, known: list[str]) -> list[str]:
        """Return an ordered sequence of concepts to learn `target` given what is already known."""
        tgt = self._require(target)
        known_keys = {self._require(k) for k in known if self._resolve(k)}
        all_prereqs = self.get_prerequisites(target)
        # Filter out already known concepts and add target at the end
        needed = [p for p in all_prereqs if p not in known_keys]
        if tgt not in known_keys:
            needed.append(tgt)
        # Topological sort respecting the graph edges
        return self._topological_sort(needed)

    def _topological_sort(self, keys: list[str]) -> list[str]:
        """Kahn's algorithm on the subgraph induced by `keys`."""
        key_set = set(keys)
        in_degree: dict[str, int] = {k: 0 for k in key_set}
        for k in key_set:
            for pred in self._radj.get(k, set()):
                if pred in key_set:
                    in_degree[k] += 1
        queue: deque[str] = deque(k for k, d in in_degree.items() if d == 0)
        result: list[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in self._adj.get(node, set()):
                if succ in key_set:
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        queue.append(succ)
        # Append any remaining (cycles)
        remaining = [k for k in keys if k not in result]
        return result + remaining

    def quiz_question(self, topic: str) -> dict:
        """Generate a multiple-choice question about the topic."""
        key = self._require(topic)
        concept = self._concepts[key]

        # Build question types
        question_templates = [
            self._quiz_definition,
            self._quiz_prerequisite,
            self._quiz_category,
            self._quiz_difficulty,
        ]
        # Choose based on available context
        for fn in question_templates:
            result = fn(key, concept)
            if result:
                return result
        # Fallback
        return self._quiz_definition(key, concept)  # type: ignore[return-value]

    def _quiz_definition(self, key: str, concept: Concept) -> dict:
        """Generate a 'what is X?' question."""
        # Extract first sentence of description
        first_sentence = concept.description.split(".")[0] + "."
        # Generate plausible distractors from other concepts in same category
        same_cat = [
            c for k, c in self._concepts.items() if c.category == concept.category and k != key
        ]
        wrong_concepts = random.sample(same_cat, min(3, len(same_cat)))
        if len(wrong_concepts) < 3:
            other = [c for k, c in self._concepts.items() if k != key]
            wrong_concepts += random.sample(other, 3 - len(wrong_concepts))

        correct = first_sentence
        distractors = [c.description.split(".")[0] + "." for c in wrong_concepts[:3]]
        options = [correct, *distractors]
        random.shuffle(options)
        correct_index = options.index(correct)

        return {
            "question": f"Which best describes **{concept.name}**?",
            "options": options,
            "correct_index": correct_index,
            "correct_answer": correct,
            "explanation": concept.description[:300],
            "topic": key,
            "difficulty": concept.difficulty,
        }

    def _quiz_prerequisite(self, key: str, concept: Concept) -> dict | None:
        """Generate a 'what is a prerequisite of X?' question."""
        preds = list(self._radj.get(key, set()))
        if not preds:
            return None
        correct_key = random.choice(preds)
        correct_name = self._concepts[correct_key].name

        # Pick 3 wrong answers from non-prerequisites
        non_preds = [k for k in self._concepts if k not in preds and k != key]
        wrong_keys = random.sample(non_preds, min(3, len(non_preds)))
        wrong_names = [self._concepts[k].name for k in wrong_keys]

        options = [correct_name] + wrong_names[:3]
        random.shuffle(options)
        correct_index = options.index(correct_name)

        return {
            "question": f"Which concept is a **direct prerequisite** of {concept.name}?",
            "options": options,
            "correct_index": correct_index,
            "correct_answer": correct_name,
            "explanation": (
                f"{correct_name} must be understood before {concept.name} because "
                f"{concept.name} builds on ideas from {correct_name}."
            ),
            "topic": key,
            "difficulty": concept.difficulty,
        }

    def _quiz_category(self, key: str, concept: Concept) -> dict:
        """Generate a 'what category does X belong to?' question."""
        correct = concept.category
        all_cats = list({c.category for c in self._concepts.values()} - {correct})
        wrong_cats = random.sample(all_cats, min(3, len(all_cats)))
        options = [correct, *wrong_cats]
        random.shuffle(options)
        return {
            "question": f"Which **field/category** does {concept.name} belong to?",
            "options": options,
            "correct_index": options.index(correct),
            "correct_answer": correct,
            "explanation": (
                f"{concept.name} is categorised under '{correct}'. "
                f"{concept.description.split('.')[0]}."
            ),
            "topic": key,
            "difficulty": "beginner",
        }

    def _quiz_difficulty(self, key: str, concept: Concept) -> dict:
        """Generate a 'what difficulty level is X?' question."""
        correct = concept.difficulty
        options = ["beginner", "intermediate", "advanced"]
        return {
            "question": f"What is the difficulty level of **{concept.name}**?",
            "options": options,
            "correct_index": options.index(correct),
            "correct_answer": correct,
            "explanation": (
                f"{concept.name} is considered '{correct}' level. "
                + (
                    "It has no prerequisites and is accessible to newcomers."
                    if correct == "beginner"
                    else "It requires foundational knowledge before studying it."
                    if correct == "intermediate"
                    else "It requires solid mastery of prerequisite topics."
                )
            ),
            "topic": key,
            "difficulty": "beginner",
        }


# Module-level singleton
_graph: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Return the singleton KnowledgeGraph instance."""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph()
    return _graph
