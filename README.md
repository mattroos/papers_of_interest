# papers_of_interest
Interesting papers I've read or ideally will read, given enough time.

## General AI

__The Seven Tools of Causal Inference, with Reflections on Machine Learning__<br/>
_Judea Pearl_<br/>
2019 https://dl.acm.org/citation.cfm?id=3241036
<details>
<summary>Intro paragraph</summary>
The dramatic success in machine learning has led to an explosion of artificial intelligence (AI) applications and increasing expectations for autonomous systems that exhibit human-level intelligence. These expectations have, however, met with fundamental obstacles that cut across many application areas. One such obstacle is adaptability, or robustness. Machine learning researchers have noted current systems lack the ability to recognize or react to new circumstances they have not been specifically programmed or trained for.
</details>


__Homeostasis and soft robotics in the design of feeling machines__<br/>
_Kingson Man, Antonio Damasio<br/>
2019 https://www.nature.com/articles/s42256-019-0103-7
<details>
<summary>Abstract</summary>
Attempts to create machines that behave intelligently often conceptualize intelligence as the ability to achieve goals, leaving unanswered a crucial question: whose goals? In a dynamic and unpredictable world, an intelligent agent should hold its own meta-goal of self-preservation, like living organisms whose survival relies on homeostasis: the regulation of body states aimed at maintaining conditions compatible with life. In organisms capable of mental states, feelings are a mental expression of the state of life in the body and play a critical role in regulating behaviour. Our goal here is to inquire about conditions that would potentially allow machines to care about what they do or think. Under certain conditions, machines capable of implementing a process resembling homeostasis might also acquire a source of motivation and a new means to evaluate behaviour, akin to that of feelings in living organisms. Drawing on recent developments in soft robotics and multisensory abstraction, we propose a new class of machines inspired by the principles of homeostasis. The resulting machines would (1) exhibit equivalents to feeling; (2) improve their functionality across a range of environments; and (3) constitute a platform for investigating consciousness, intelligence and the feeling process itself.
</details>


## Deep learning

__Learning Algorithmic Solutions to Symbolic Planning Tasks with a Neural Computer__<br/>
_Daniel Tanneberg, Elmar Rueckert, Jan Peters_<br/>
2019 https://arxiv.org/abs/1911.00926
<details>
<summary>Abstract</summary>
A key feature of intelligent behavior is the ability to learn abstract
strategies that transfer to unfamiliar problems. Therefore, we present a novel
architecture, based on memory-augmented networks, that is inspired by the von
Neumann and Harvard architectures of modern computers. This architecture
enables the learning of abstract algorithmic solutions via Evolution Strategies
in a reinforcement learning setting. Applied to Sokoban, sliding block puzzle
and robotic manipulation tasks, we show that the architecture can learn
algorithmic solutions with strong generalization and abstraction: scaling to
arbitrary task configurations and complexities, and being independent of both
the data representation and the task domain.
</details>


## Spiking neural networks

__Towards Scalable, Efficient and Accurate Deep Spiking Neural Networks with Backward Residual Connections, Stochastic Softmax and Hybridization__<br/>
_Priyadarshini Panda, Aparna Aketi, and Kaushik Roy_<br/>
2019 https://arxiv.org/abs/1910.13931
<details>
<summary>Abstract</summary>
Spiking Neural Networks (SNNs) may offer an energy-efficient alternative for
implementing deep learning applications. In recent years, there have been
several proposals focused on supervised (conversion, spike-based gradient
descent) and unsupervised (spike timing dependent plasticity) training methods
to improve the accuracy of SNNs on large-scale tasks. However, each of these
methods suffer from scalability, latency and accuracy limitations. In this
paper, we propose novel algorithmic techniques of modifying the SNN
configuration with backward residual connections, stochastic softmax and hybrid
artificial-and-spiking neuronal activations to improve the learning ability of
the training methodologies to yield competitive accuracy, while, yielding large
efficiency gains over their artificial counterparts. Note, artificial
counterparts refer to conventional deep learning/artificial neural networks.
Our techniques apply to VGG/Residual architectures, and are compatible with all
forms of training methodologies. Our analysis reveals that the proposed
solutions yield near state-of-the-art accuracy with significant
energy-efficiency and reduced parameter overhead translating to hardware
improvements on complex visual recognition tasks, such as, CIFAR10, Imagenet
datatsets.
</details>


## Neuroscience-inspired neural networks

__Working memory facilitates reward-modulated Hebbian learning in recurrent neural networks__<br/>
_Roman Pogodin, Dane Corneil, Alexander Seeholzer, Joseph Heng, Wulfram Gerstner_<br/>
2019 https://arxiv.org/abs/1910.10559
<details>
<summary>Abstract</summary>
Reservoir computing is a powerful tool to explain how the brain learns
temporal sequences, such as movements, but existing learning schemes are either
biologically implausible or too inefficient to explain animal performance. We
show that a network can learn complicated sequences with a reward-modulated
Hebbian learning rule if the network of reservoir neurons is combined with a
second network that serves as a dynamic working memory and provides a
spatio-temporal backbone signal to the reservoir. In combination with the
working memory, reward-modulated Hebbian learning of the readout neurons
performs as well as FORCE learning, but with the advantage of a biologically
plausible interpretation of both the learning rule and the learning paradigm.
</details>

__CTNN: Corticothalamic-inspired neural network__<br/>
_Leendert A Remmelzwaal, Amit Mishra, George F R Ellis_<br/>
2019 https://arxiv.org/abs/1910.12492
<details>
<summary>Abstract</summary>
Sensory predictions by the brain in all modalities take place as a result of
bottom-up and top-down connections both in the neocortex and between the
neocortex and the thalamus. The bottom-up connections in the cortex are
responsible for learning, pattern recognition, and object classification, and
have been widely modelled using artificial neural networks (ANNs). Current
neural network models (such as predictive coding models) have poor processing
efficiency, and are limited to one input type, neither of which is
bio-realistic. Here, we present a neural network architecture modelled on the
corticothalamic connections and the behaviour of the thalamus: a
corticothalamic neural network (CTNN). The CTNN presented in this paper
consists of an auto-encoder connected to a difference engine, which is inspired
by the behaviour of the thalamus. We demonstrate that the CTNN is input
agnostic, multi-modal, robust during partial occlusion of one or more sensory
inputs, and has significantly higher processing efficiency than other
predictive coding models, proportional to the number of sequentially similar
inputs in a sequence. This research helps us understand how the human brain is
able to provide contextual awareness to an object in the field of perception,
handle robustness in a case of partial sensory occlusion, and achieve a high
degree of autonomous behaviour while completing complex tasks such as driving a
car.
</details>

## Neuroevolution, artificial life and open-ended evolution

__Convolution by Evolution: Differentiable Pattern Producing Networks__<br/>
_Chrisantha Fernando, Dylan Banarse, Malcolm Reynolds et al._<br/>
2016 https://dl.acm.org/citation.cfm?id=2908890
<details>
<summary>Abstract</summary>
In this work we introduce a differentiable version of the Compositional Pattern Producing Network, called the DPPN. Unlike a standard CPPN, the topology of a DPPN is evolved but the weights are learned. A Lamarckian algorithm, that combines evolution and learning, produces DPPNs to reconstruct an image. Our main result is that DPPNs can be evolved/trained to compress the weights of a denoising autoencoder from 157684 to roughly 200 parameters, while achieving a reconstruction accuracy comparable to a fully connected network with more than two orders of magnitude more parameters. The regularization ability of the DPPN allows it to rediscover (approximate) convolutional network architectures embedded within a fully connected architecture. Such convolutional architectures are the current state of the art for many computer vision applications, so it is satisfying that DPPNs are capable of discovering this structure rather than having to build it in by design. DPPNs exhibit better generalization when tested on the Omniglot dataset after being trained on MNIST, than directly encoded fully connected autoencoders. DPPNs are therefore a new framework for integrating learning and evolution.
</details>


## Neuroscience

__The Cell-Type Specific Cortical Microcircuit: Relating Structure and Activity in a Full-Scale Spiking Network Model__<br/>
_Tobias C. Potjans, Markus Diesmann_<br/>
2014 https://academic.oup.com/cercor/article/24/3/785/398560
<details>
<summary>Abstract</summary>
In the past decade, the cell-type specific connectivity and activity of local cortical networks have been characterized experimentally to some detail. In parallel, modeling has been established as a tool to relate network structure to activity dynamics. While available comprehensive connectivity maps ( Thomson, West, et al. 2002; Binzegger et al. 2004) have been used in various computational studies, prominent features of the simulated activity such as the spontaneous firing rates do not match the experimental findings. Here, we analyze the properties of these maps to compile an integrated connectivity map, which additionally incorporates insights on the specific selection of target types. Based on this integrated map, we build a full-scale spiking network model of the local cortical microcircuit. The simulated spontaneous activity is asynchronous irregular and cell-type specific firing rates are in agreement with in vivo recordings in awake animals, including the low rate of layer 2/3 excitatory cells. The interplay of excitation and inhibition captures the flow of activity through cortical layers after transient thalamic stimulation. In conclusion, the integration of a large body of the available connectivity data enables us to expose the dynamical consequences of the cortical microcircuitry.
</details>

__Cortical credit assignment by Hebbian, neuromodulatory and inhibitory
  plasticity__<br/>
_Johnatan Aljadeff, James D'amour, Rachel E. Field, Robert C. Froemke, Claudia Clopath_<br/>
2019 https://arxiv.org/abs/1911.00307
<details>
<summary>Abstract</summary>
The cortex learns to make associations between stimuli and spiking activity
which supports behaviour. It does this by adjusting synaptic weights. The
complexity of these transformations implies that synapses have to change
without access to the full error information, a problem typically referred to
as "credit-assignment". However, it remains unknown how the cortex solves this
problem. We propose that a combination of plasticity rules, 1) Hebbian, 2)
acetylcholine-dependent and 3) noradrenaline-dependent excitatory plasticity,
together with 4) inhibitory plasticity restoring E/I balance, effectively
solves the credit assignment problem. We derive conditions under-which a neuron
model can learn a number of associations approaching its theoretical capacity.
We confirm our predictions regarding acetylcholine-dependent and inhibitory
plasticity by reanalysing experimental data. Our work suggests that detailed
cortical E/I balance reduces the dimensionality of the problem of associating
inputs with outputs, thereby allowing imperfect "supervision" by
neuromodulatory systems to guide learning effectively.
</details>

