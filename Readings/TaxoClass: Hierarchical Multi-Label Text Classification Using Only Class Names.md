제공해주신 PDF 파일인 "TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names"의 모든 내용을 Markdown 형식으로 정리했습니다. 논문의 구조, 수식, 표, 참고문헌을 원문 그대로 포함했습니다.

***

# [cite_start]TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names [cite: 1]

**Jiaming Shen, Wenda Qiu, Yu Meng, Jingbo Shang, Xiang Ren, Jiawei Han**
University of Illinois at Urbana-Champaign, IL, USA, University of California, San Diego, CA, USA
University of Southern California, CA, USA
(js2, qiuwenda, yumeng5, hanj) [cite_start]@illinois.edu, jshang@ucsd.edu, xiangren@ucs.edu [cite: 2, 3, 4, 5, 6]

## Abstract
[cite_start]Hierarchical multi-label text classification (HMTC) aims to tag each document with a set of classes from a class hierarchy. [cite: 8] [cite_start]Most existing HMTC methods train classifiers using massive human-labeled documents, which are often too costly to obtain in real-world applications. [cite: 9] [cite_start]In this paper, we explore to conduct HMTC based on only class surface names as supervision signals. [cite: 10] [cite_start]We observe that to perform HMTC, human experts typically first pinpoint a few most essential classes for the document as its "core classes", and then check core classes' ancestor classes to ensure the coverage. [cite: 11] [cite_start]To mimic human experts, we propose a novel HMTC framework, named TaxoClass. [cite: 12] [cite_start]Specifically, TaxoClass (1) calculates document-class similarities using a textual entailment model, (2) identifies a document's core classes and utilizes confident core classes to train a taxonomy-enhanced classifier, and (3) generalizes the classifier via multi-label self-training. [cite: 13] [cite_start]Our experiments on two challenging datasets show TaxoClass can achieve around 0.71 Example-F1 using only class names, outperforming the best previous method by 25%. [cite: 14]

## 1. Introduction
[cite_start]Hierarchical multi-label text classification (HMTC) aims to assign each text document to a set of relevant classes from a class taxonomy. [cite: 16] [cite_start]As a fundamental task in NLP, HMTC has many applications such as product categorization (Goumy and Mejri, 2018), semantic indexing (Li et al., 2019), and fine-grained entity typing (Xu and Barbosa, 2018). [cite: 17]

[cite_start]Most existing methods address HMTC in a supervised fashion they first ask humans to provide many labeled documents and then train a text classifier for prediction. [cite: 18] [cite_start]Many classifiers have been developed with different deep learning architectures such as CNN (Kim, 2014), RNN (You et al., 2019), Attention Network (Huang et al., 2019), and achieved decent performance when trained on massive human-labeled documents. [cite: 19] [cite_start]Despite such a success, people find that applying these methods to many real-world scenarios remains challenging as the human labeling process is often too time-consuming and expensive. [cite: 20, 37]

[cite_start]Recently, more studies have been developed to address text classification using smaller amount of labeled data. [cite: 38] [cite_start]First, several semi-supervised methods (Gururangan et al., 2019; Berthelot et al., 2019) propose to use abundant unlabeled documents to assist model training on labeled dataset. [cite: 39] [cite_start]Although mitigating the human annotation burden, these methods still require a labeled dataset that covers all classes, which could be too expensive to obtain when we have a large number of classes in HMTC. [cite: 40] [cite_start]Second, some weakly-supervised models exploit class indicative keywords (Meng et al., 2018; Zeng et al., 2019; Mekala and Shang, 2020) or class surface names (Meng et al., 2020; Wang et al., 2020) to derive pseudo-labeled data for model training. [cite: 41] [cite_start]Nevertheless, these models all assume each document has only one class and all class surface names (or class indicative keywords) must appear in the corpus, which are too restrictive for HMTC. [cite: 42]

[cite_start]In this paper, we study the problem of weakly-supervised hierarchical multi-label text classification where only class surface names, a class taxonomy, and an unlabeled corpus are available for model training. [cite: 45] [cite_start]This setting is closer to how humans resolve the HMTC problem we perform classification by understanding each class from its surface name rather than learning from labeled documents. [cite: 46] [cite_start]We observe that when asked to assign multiple classes to a document, humans will first pinpoint most essential "core classes" and then check whether their ancestor classes in the taxonomy should also be tagged. [cite: 47] [cite_start]Taking the document in Fig. 1 as an example, humans can quickly identify this review text is clearly about "baby cereal" and "crafted cereal", which are the core classes. [cite: 48] [cite_start]After assigning these two most essential classes to the document, people continue to check the core classes ancestor classes and find "feeding" as well as "baby food" should be tagged. [cite: 49]

[cite_start]Motivated by the above human labeling process, we propose TaxoClass, a weakly-supervised HMTC framework including four major steps. [cite: 50] [cite_start]First, we calculate the document-class similarity using a pre-trained textual entailment model (Yin et al., 2019). [cite: 51] [cite_start]Second, we identify each document's core classes by (1) selecting candidate core classes that are most similar to the document at each level in a top-down fashion, and (2) choosing (document, candidate core class) pairs that are salient across the whole unlabeled corpus. [cite: 52] [cite_start]Third, we derive training data from document core classes and use them to train a text classifier. [cite: 53] [cite_start]This classifier includes a document encoder based on pre-trained BERT (Devlin et al., 2019), a class encoder capturing class taxonomy structure, and a text matching network computing the probability of a document being tagged with each class. [cite: 54] [cite_start]Finally, we generalize this text classifier using multi-label self-training on all unlabeled documents. [cite: 55]

**Contributions.** To summarize, our major contributions are as follows:
(1) [cite_start]We propose a weakly-supervised framework TaxoClass that only requires class surface names to perform hierarchical multi-label text classification. [cite: 56] [cite_start]To the best of our knowledge, TaxoClass is the first weakly-supervised HMTC method. [cite: 57]
(2) [cite_start]We develop an unsupervised method to identify document core classes based on which a text classifier can be learned. [cite: 58]
(3) [cite_start]We conduct extensive experiments to verify the effectiveness of TaxoClass on two real-world datasets. [cite: 59]

## 2. Problem Formulation
[cite_start]In this section, we introduce the notations and present our task definition. [cite: 61]

[cite_start]**Notations.** A corpus $\mathcal{D}=\{D_{1},...,D_{N}\}$ is a text collection where each document $D_{i}\in\mathcal{D}$ is a sequence of words. [cite: 62] [cite_start]A class taxonomy $\mathcal{T}=(\mathcal{C},\mathcal{R})$ is a directed acyclic graph where each node represents a class $c_{j}$ and each directed edge $\langle c_{m},c_{n}\rangle\in\mathcal{R}$ indicates that parent class $c_{m}$ is more general than the child class $c_{n}$. [cite: 63] [cite_start]In this work, we assume each class $c_{j}$ has a surface name $s_{j}$ (either a word or a phrase) that serves as the weak supervision signal. [cite: 64]

[cite_start]**Task Definition.** Given an unlabeled corpus D, a class hierarchy $\mathcal{T}=(\mathcal{C},\mathcal{R})$, and class surface names $\mathcal{S}=\{s_{j}\}_{j=1}^{|\mathcal{C}|}$ our task is to learn a text classifier $f(\cdot)$ that maps a new document $D_{new}$ to its target $y=[y_{1},...,y_{|\mathcal{C}|}]\in\mathcal{Y}=\{0,1\}^{|\mathcal{C}|}$ where $y_{j}$ equals to 1 if this document is categorized with class $c_{j}$ and 0 otherwise. [cite: 65]

[cite_start]**Discussion.** When the number of classes C is large (as it is in many HMTC applications), we can no longer assume all class surface names in S will explicitly appear in the given corpus D as done in most previous studies (Meng et al., 2019; Li et al., 2019; Wang et al., 2020). [cite: 66] [cite_start]This is because many class names are actually summarizing phrases provided by humans (e.g., "grocery & gourmet food" in Fig. 1). [cite: 67] [cite_start]As a result, we need to design a method that works under such a scenario. [cite: 68]

## 3. Our TaxoClass Framework
[cite_start]Our TaxoClass framework consists of four major steps: (1) document-class similarity calculation, (2) document core class mining, (3) core class guided classifier training, and (4) multi-label self-training. [cite: 70] [cite_start]Fig. 2 shows our framework overview and below sections discuss each step in more details. [cite: 71]

### 3.1 Document-Class Similarity Calculation
[cite_start]We take a textual entailment approach (Yin et al., 2019) to calculate the semantic similarity between each (document, class) pair. [cite: 72] [cite_start]This approach imitates how humans determine whether a document is similar to a class or not we read this document, create a hypothesis by filling the class name into a template (e.g., "this document is about _"), and ask ourselves to what extent this hypothesis is correct, given the context document. [cite: 73]

[cite_start]In this work, we adopt a pre-trained textual entailment model that inputs a document $D_{i}$ as the "premise", a template filled with a class name $s_{j}$ as the "hypothesis", and outputs a probability of how likely this premise can entail the hypothesis. [cite: 74, 127] [cite_start]We treat this probability $P(D_{i}\rightarrow c_{j})$ as the document-class similarity $sim(D_{i},c_{j})$. [cite: 128] [cite_start]More specifically, we use **RoBERTa-Large-MNLI** as our textual entailment model which utilizes the pre-trained Roberta-Large as its backbone and is fine-tuned on the MNLI dataset. [cite: 129]

### 3.2 Document Core Class Mining
[cite_start]When asked to tag a document with a set of classes from a class taxonomy, humans will first pinpoint a few classes that are most essential to this document. [cite: 131] [cite_start]We refer to those most essential classes as the "core classes" and identify them in below two steps. [cite: 132]

#### 3.2.1 Core Class Candidate Selection
[cite_start]We observe that on average each document is tagged with a small set of classes from the entire class taxonomy. [cite: 134] [cite_start]Therefore, we first reduce the search space of core classes using a top-down approach (c.f. Fig. 3). [cite: 135] [cite_start]Given a document D, we start with the "Root" class at level $l=0$ find its two children classes that have the highest similarity with D, and add them into a queue. [cite: 136] [cite_start]Then, for each class at level $l$ in the queue, we select $l+2$ classes from its children classes that are most similar to D. After all level $l$ classes are processed, we aggregate all selected children classes and choose $(l+1)^{2}$ classes (at level $l+1$) with the highest path score ($ps$) defined below: [cite: 137, 139]

$$ps(c) = \max_{c_k \in Par(c)} \{ps(c_k) \cdot sim(c, D)\}, \quad ps(Root) = 1, \quad (1)$$

[cite_start]where $Par(c_{j})$ is class $c_{j}$ parent class set. [cite: 140-144] [cite_start]All chosen classes (at level $l+1$) will be pushed into the queue and we stop this process when no class in the queue has further children. [cite: 144] [cite_start]Finally, all classes that have entered the queue, except for the "Root" class, consist of the core class candidate set. [cite: 145] [cite_start]We use $\mathbb{C}_{i}^{cand}$ to denote the candidate core class set of document $D_{i}$. [cite: 146]

#### 3.2.2 Confident Core Class Identification
[cite_start]For each document, we identify its core classes from the above selected candidate set based on two observations. [cite: 148] [cite_start]First, a document usually has higher similarity with its core class $c$ than with the parent and sibling classes of $c$. [cite: 149] [cite_start]Take the document $D_{2}$ in Fig. 2 as an example, the similarity between $D_{2}$ and its core class "crib" is 0.95, much higher than the similarity between $D_{2}$ and core class's parent class "nursery" (0.6) as well as core class's sibling classes. [cite: 150] [cite_start]Based on this observation, we define the "confidence score" of a candidate core class $c$ for a document $D$ as below: [cite: 151]

$$conf (D, c) = sim(D,c) - \max_{c' \in Par(c) \cup Sib(c)} \{sim(D,c')\}, \quad (2)$$

[cite_start]where $Sib(c)$ represents the sibling class set of $c$. [cite: 153-154]

[cite_start]Our second observation is that the similarity between a document $D$ and its core class $c$ is salient from a corpus-wise perspective. [cite: 155] [cite_start]Namely, if a class $c$ is a document $D$'s core class, the confidence score $conf(D,c)$ is higher than the median confidence score between class $c$ and all documents tagged with $c$ (denoted as $\mathcal{D}(c))$. [cite: 156, 181] Formally, we have:

$$conf(D, c) > \text{median}\{conf (D',c) \mid D' \in \mathcal{D}(c)\}. \quad (3)$$

[cite_start]According to this observation, we check each class in document $D_{i}$ candidate core set $\mathbb{C}_{i}^{cand}$ and add classes that satisfy the above criteria into the final core class set $\mathbb{C}_{i}$. [cite: 183] [cite_start]Note here this core class set $\mathbb{C}_{i}$ could be empty when document $D_{i}$ does not have any confident core class. [cite: 184]

### 3.3 Core Class Guided Classifier Training
[cite_start]Based on identified document core classes, we train one classifier for hierarchical multi-label text classification. [cite: 185] [cite_start]Below we first introduce our classifier architecture and then present our training method. [cite: 186]

#### 3.3.1 Text Classifier Architecture
[cite_start]We design our classifier to have a dual-encoder architecture: one document encoder maps document $D_{i}$ to its representation $D_{i}$, one class encoder learns class $c_{j}$ representation $c_{j}$, and one matching network returns the probability of document $D_{i}$ being tagged with class $c_{j}$. [cite: 187]

[cite_start]**Document Encoder.** In this work, we instantiate our document encoder $g_{doc}(\cdot)$ to be a pre-trained BERT-base-uncased model (Devlin et al., 2019) and follow previous work (Chang et al., 2019; Meng et al., 2020) to use the [CLS] token representation as the document representation. [cite: 188]

[cite_start]**Class Encoder.** For class encoder $g_{class}(\cdot)$, we follow (Shen et al., 2020) and use a graph neural network (GNN) (Kipf and Welling, 2017) to model the class taxonomy structure. [cite: 189] [cite_start]This taxonomy-enhanced class encoder can capture both the textual information from class surface names and structural information from the class taxonomy. [cite: 190] [cite_start]Given a class $c_{j}$, we first obtain its ego network that includes its parent and children classes in the class taxonomy, as shown in Fig. 4. Then, we input this ego network to a GNN that propagates node features over the network structure. [cite: 191, 194] [cite_start]The node features are initialized with the pre-trained word embeddings of class surface names. [cite: 194] [cite_start]The propagation mechanism updates the feature of a node $u$ by iteratively aggregating representations of its neighbors and itself. [cite: 195] Formally, we define a GNN with $L$-layers as follows:

$$h_{u}^{(l)}=\text{ReLU}(\sum_{v\in N(u)}\alpha_{uv}^{(l-1)}W^{(l-1)}h_{v}^{(l-1)}) \quad (4)$$

[cite_start]where $l\in\{1,...,L\}$, $N(u)$ includes node $u$'s neighbors and itself, $\alpha_{uv}^{(l-1)}=\frac{1}{\sqrt{|N(u)||N(v)|}}$ is a normalization constant (same for all layers), and $W^{(l-1)}$ are learnable parameters. [cite: 197-200]

After obtaining individual node features, we combine them into a vector representing the whole ego network $\mathcal{G}$ as follows:

$$c = \sum_{u \in \mathcal{G}} h_u^{(L)} \quad (5)$$

[cite_start]As this ego network is centered on class $c$ and encodes its both textual and structural information, we treat this final graph representation as the class representation $c_{j}$. [cite: 205]

[cite_start]**Text Matching Network.** Based on the document representation $D_{i}$ and the class representation $c_{j}$, we use a log-bilinear text matching model to compute the probability of document $D_{i}$ being tagged with class $c_{j}$ as follows: [cite: 206]

$$p_{ij}=P(y_{j}=1|D_{i})=\sigma(\exp(c_{j}^{T}BD_{i})), \quad (6)$$

[cite_start]where $\sigma(\cdot)$ is the sigmoid function and B is a learnable interaction matrix. [cite: 207-208]

#### 3.3.2 Text Classifier Training
[cite_start]We use our discovered document confident core classes to train a text classifier. [cite: 210] [cite_start]One intuitive strategy is to treat each document's core classes as positive classes and all the remaining classes as negative classes. [cite: 211] [cite_start]However, this strategy has a high false negative rate because some non-core classes could still be relevant to the document (c.f. Fig. 1). [cite: 212, 230] [cite_start]We observe a document's multiple labeled classes usually have some ancestor-descendent relations in the class hierarchy $\mathcal{T}=(\mathcal{C},\mathcal{R})$. [cite: 231] [cite_start]This implies that given a document's core class, its parent class and some of its children classes are also likely to be tagged with this document. [cite: 232] [cite_start]Therefore, we introduce all core classes' parent classes into the positive class set and exclude their children classes from the negative class set. [cite: 233] Formally, given a document $D_{i}$ with its core class set $\mathbb{C}_{i}$, we define its positive and negative class set as follows:

$$\mathbb{C}_{i}^{pos}=(\bigcup_{c_{j}\in\mathbb{C}_{i}}Par(c_{j}))\cup\mathbb{C}_{i}, \quad \mathbb{C}_{i}^{neg}=\mathcal{C}-\mathbb{C}_{i}^{pos}-\bigcup_{c_{j}\in\mathbb{C}_{i}}Chd(c_{j}), \quad (7)$$

[cite_start]where $Chd(c_{j})$ is class $c_{j}$ children class set. [cite: 235, 237] Finally, we train our classification model using the below binary cross entropy (BCE) loss:

$$\mathcal{L}=-\sum_{D_i \in \mathcal{D}, \mathbb{C}_i \neq \emptyset} (\sum_{c_{j}\in\mathbb{C}_{i}^{pos}}\log p_{ij}+\sum_{c_{j}\in \mathbb{C}_{i}^{neg}}\log(1-p_{ij})), \quad (8)$$

[cite_start]where we exclude the documents without any confident core class from the loss calculation. [cite: 238-239]

### 3.4 Multi-label Self-Training
[cite_start]After training the text classifier based on document core classes, we propose to further refine the model via self-training on the entire unlabeled corpus D for better generalization. [cite: 241] The idea of self-training (ST) (Xie et al., 2016) is to iteratively use the model's current prediction P to compute a target distribution Q which guides the model for refinement. [cite_start]In general, the ST objective is expressed with the KL divergence loss as below: [cite: 242, 247]

$$\mathcal{L}_{ST}=KL(Q||P)=\sum_{i=1}^{|\mathcal{D}|}\sum_{j=1}^{|\mathcal{C}|}q_{ij}\log\frac{q_{ij}}{p_{ij}} \quad (9)$$

[cite_start]The target distribution is constructed by enhancing high-confidence predictions while down-weighting low-confidence ones: [cite: 249]

$$t_{ij}=\frac{p_{ij}^{2}/(\sum_{i}p_{ij})}{p_{ij}^{2}/(\sum_{i}p_{ij})+(1-p_{ij})^{2}/(\sum_{i}(1-p_{ij}))}. \quad (10)$$

[cite_start]Different from the previous studies (Meng et al., 2018; Yu et al., 2020), our target distribution Q can be applied to multi-label classification problem as it normalizes the current predictions P for each individual class. [cite: 251] [cite_start]Intuitively, this equation can enhance high-confidence predictions while down-weighting low-confidence predictions. [cite: 252] [cite_start]This is because if example $i$ is more confidently labeled with class $j$ than other examples, we will have a large $p_{ij}$ that dominates the $\sum_{i}p_{ij}$ term. [cite: 253] [cite_start]Consequently, Eq 10 computes a large $q_{ij}$, which further pushes the model to predict class $j$ for example $i$. [cite: 254] In practice, instead of updating the target distribution for every training example, we update it every 25 batches and train the model with Eq. (9)[cite_start], which makes the self-training process more efficient and robust. [cite: 255-256]

[cite_start]**Algorithm 1: TaxoClass Framework.** [cite: 215]
[cite_start]Input: An unlabeled corpus D, a class taxonomy T with class names S, an entailment model M, total number of batches B. [cite: 216]
[cite_start]Output: A trained classifier $f(\cdot)$. [cite: 217]
1. [cite_start]Use model M to compute document-class similarity (c.f. Sect. 3.1); [cite: 218]
2. [cite_start]Obtain document core classes $\{(D_{i},\mathbb{C}_{i})|D_{i}\in\mathcal{D}\}$ (c.f. Sect. 3.2); [cite: 219-220]
3. Train classifier $f(\cdot)$ with Eq. (8)[cite_start]; [cite: 221]
4. [cite_start]for $i$ from 1 to B do [cite: 222]
5.    if $i$ mod $25=0$ then Update Q with Eq. (10)[cite_start]; [cite: 226-227]
6.    Train classifier $f(\cdot)$ with Eq. (9)[cite_start]; [cite: 228]
7. [cite_start]Return $f(\cdot)$ [cite: 229]

## 4. Experiments

### 4.1 Datasets
[cite_start]We use two public datasets from different domains to evaluate our method: (1) **Amazon-531** (McAuley and Leskovec, 2013) contains 49,145 product reviews and a three-level class taxonomy consisting of 531 classes; and (2) **DBPedia-298** (Lehmann et al., 2015) includes 245,832 Wikipedia articles and a three-level class taxonomy with 298 classes. [cite: 259-260, 263] [cite_start]Documents in both datasets are lower-cased and truncated to has maximum 500 tokens. [cite: 264] [cite_start]We list the data statistics in Table 1. [cite: 265]

[cite_start]**Table 1: Dataset statistics.** [cite: 244]

| Dataset | #Train | #Test | #Classes |
| :--- | :--- | :--- | :--- |
| Amazon-531 | 29,487 | 19,685 | 531 |
| DBPedia-298 | 196,665 | 49,167 | 298 |

Supervised methods are trained on the entire training set. Weakly-supervised methods are trained by treating the training set as unlabeled data. [cite_start]All methods are evaluated on the test set. [cite: 244-246]

### 4.2 Compared Methods
[cite_start]To the best of our knowledge, we are the first to study weakly-supervised HMTC problem and there is no directly comparable baseline under the exact same setting as ours. [cite: 267] [cite_start]Therefore, we choose a wide range of representative methods that are most related to TaxoClass and adapt them to our problem setting, described as follows. [cite: 268]

* [cite_start]**Hier-doc2vec** (Le and Mikolov, 2014): This weakly-supervised method first embeds documents and classes into a shared semantic space, and then recursively selects the class of the highest embedding similarity with the document in a top-down fashion. [cite: 269] [cite_start]We set the embedding dimensionality to be 100 and use the default value for all other hyper-parameters. [cite: 270]
* [cite_start]**WeSHClass** (Meng et al., 2019): Another weakly-supervised method that generates pseudo documents to pre-train a text classifier and bootstraps the pre-trained classifier on unlabeled documents with self-training. [cite: 271] [cite_start]The class surface names are treated as the "class-related keywords" in this method. [cite: 272] [cite_start]For the pseudo document generation step, we use its internal LSTM language model. [cite: 273] [cite_start]We treat all classes in its returned class path as the output classes. [cite: 274]
* [cite_start]**SS-PCEM** (Xiao et al., 2019): This semi-supervised method uses a generative model to generate documents based on a class path sampled from the class taxonomy. [cite: 275] [cite_start]Both labeled and unlabeled documents are used to fit this generative model via the EM algorithm. [cite: 276] [cite_start]Finally, it uses the posterior probability of a test document to predict its labeled classes. [cite: 277] [cite_start]Among different base classifiers, we choose their author reported best variant PCEM in this study. [cite: 278] [cite_start]We use 30% of labeled training documents for this method. [cite: 279]
* [cite_start]**Hier-0Shot-TC** (Yin et al., 2019): This zero-shot method uses a pre-trained textual entailment model to predict to what extent a document (as the premise text) can entail a template filled with the class name (as the hypothesis text). [cite: 284] [cite_start]Similar to Hier-doc2vec, we select the class with the highest entailment score at each level in a top-down recursive fashion. [cite: 285] [cite_start]For fair comparison, we change its internal BERT-base-uncased model to RoBERTa-large-mnli model as is used in our method. [cite: 286]
* [cite_start]**TaxoClass**: Our proposed weakly-supervised framework that identifies document core classes, leverages core classes to train a taxonomy-enhanced text classifier, and generalizes the classifier using multi-label self-training. [cite: 287] [cite_start]We also evaluate two ablations: **TaxoClass-NoST** which removes the multi-label self-training step, and **TaxoClass-NoGNN** which replaces the GNN-based class encoder with a simple embedding layer initialized with pre-trained word embeddings. [cite: 288]

### 4.3 Evaluation Metrics
[cite_start]We follow previous studies (Partalas et al., 2015; Prabhu et al., 2018) and evaluate the multi-label classification results from different aspects using various metrics. [cite: 290] [cite_start]The first metric is **Example-F1** which calculates the average F1 scores for all documents as follows: [cite: 291]

$$\text{Example-F1} =\frac{1}{N}\sum_{i=1}^{N}\frac{2|\mathbb{C}_{i}^{true}\cap\mathbb{C}_{i}^{pred}|}{|\mathbb{C}_{i}^{true}|+|\mathbb{C}_{i}^{pred}|}$$

[cite_start]where $\mathbb{C}_{i}^{true}(\mathbb{C}_{i}^{pred})$ is the true (model predicted) class set of document $D_{i}$. [cite: 292-293]

[cite_start]Moreover, as many applications formalize the HMTC as a class ranking problem (Jain et al., 2016; Guo et al., 2019), we convert predicted class set $\mathbb{C}_{i}^{pred}$ into a rank list $\mathbb{R}_{i}^{pred}$ based on each class's model predicted probability and calculate **Precision at k (P@k)** as follows: [cite: 294]

$$P@k=\frac{1}{N}\sum_{i=1}^{N}\frac{|\mathbb{C}_{i}^{true}\cap\mathbb{R}_{i,1:k}^{pred}|}{\min(k,|\mathbb{C}_{i}^{true}|)},$$

[cite_start]where $\mathbb{R}_{i,1:k}^{pred}$ is each method predicted top k most likely classes for $D_{i}$. [cite: 295, 304] [cite_start]Finally, for methods able to return the probability of a document being tagged with each class in the taxonomy, we calculate their **Mean Reciprocal Rank (MRR)** as follows: [cite: 304]

$$MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{|\mathbb{C}_{i}^{true}|}\sum_{c_{j}\in\mathbb{C}_{i}^{true}}\frac{1}{R_{ij}}$$

[cite_start]where $R_{ij}$ is the "rank" of document $D_{j}$ true class $c_{j}$ in model predicted rank list (over all classes). [cite: 305-306]

### 4.4 Experiment Settings
[cite_start]For all baseline methods except Hier-doc2vec, we use the public implementations from their authors and leave the hyper-parameters unchanged. [cite: 308] [cite_start]For both Hier-0Shot-TC and our method, we adopt the same public RoBERTa-Large-MNLI model as the textual entailment model and use the same hypothesis template: "this product is about _" for Amazon-531 dataset and "this example is _" for DBPedia-298 dataset. [cite: 309-310] [cite_start]We use AdamW optimizer to train our model with batch size 64, learning rate 5e-5 for all parameters in BERT document encoder and learning rate 4e-3 for all remaining parameters. [cite: 311] [cite_start]During the multi-label self-training stage (c.f. Sect. 3.4), we use learning rate 1e-6 for all parameters in the BERT document encoder and 5e-4 for all remaining parameters. [cite: 312] [cite_start]We run all experiments on a single cluster with 80 CPU cores and a Quadro RTX 8000 GPU. [cite: 313] [cite_start]All deep learning models are moved to the GPU for faster inference speed. [cite: 314] [cite_start]With batch size 64, the TaxoClass framework consumes about 10GB GPU memory. [cite: 315] [cite_start]In principle, all methods should be runnable on CPU. [cite: 316]

### 4.5 Overall Performance Comparison
[cite_start]Table 2 presents the overall results of all compared methods. [cite: 322] [cite_start]First, we find most weakly-supervised and zero-shot method can outperform the semi-supervised method SS-PCEM even the later has access to 30% of labeled documents. [cite: 323] [cite_start]Second, we can see that TaxoClass has the overall best performance across all the metrics and defeats the second best method by a large margin. [cite: 324] [cite_start]Comparing TaxoClass with TaxoClass-NoGNN, we show the importance of incorporating taxonomy structure into the class encoder. [cite: 325] [cite_start]Moreover, the improvement of TaxoClass over TaxoClass-NoST demonstrates the effectiveness of our multi-label self-training. [cite: 326]

[cite_start]**Table 2: Evaluation of all compared methods on two datasets.** [cite: 301-303]

| Method | Amazon-531 |||| DBPedia-298 ||||
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **Example-F1** | **P@1** | **P@3** | **MRR** | **Example-F1** | **P@1** | **P@3** | **MRR** |
| Hier-doc2vec | 0.3157 | 0.5805 | 0.3115 | N/A | 0.1443 | 0.2635 | 0.1443 | N/A |
| WeSHClass | 0.2458 | 0.5773 | 0.2517 | N/A | 0.3047 | 0.5359 | 0.3048 | N/A |
| SS-PCEM | 0.2921 | 0.5369 | 0.2948 | 0.3004 | 0.3845 | 0.7424 | 0.3845 | 0.4032 |
| Hier-0Shot-TC | 0.4742 | 0.7144 | 0.4610 | N/A | 0.6765 | 0.7871 | 0.6765 | N/A |
| **TaxoClass-NoST** | 0.5431 | 0.7918 | 0.5414 | 0.5911 | 0.7712 | 0.8621 | 0.7712 | 0.8221 |
| **TaxoClass-NoGNN** | 0.5271 | 0.7642 | 0.5213 | 0.5621 | 0.7241 | 0.8154 | 0.7241 | 0.7692 |
| **TaxoClass** | **0.5934** | **0.8120** | **0.5894** | **0.6332** | **0.8156** | **0.8942** | **0.8156** | **0.8762** |

### 4.6 Effectiveness of Core Class Mining
[cite_start]We evaluate the effectiveness of our core class mining method as follows. [cite: 328] [cite_start]First, we define a set of rival methods and use them to generate various sets of "core classes". [cite: 329] [cite_start]Then, we derive pseudo-training data for each generated core class set and use it to learn a text classifier with the same architecture as the one in TaxoClass. [cite: 330] [cite_start]Finally, we report each model's performance on the test set. [cite: 331] [cite_start]Note here we skip the self-training step to ensure the "core class based pseudo-training data" is the only variable. [cite: 332]

[cite_start]**Table 3: Evaluation of core class mining algorithms on Amazon-531 dataset.** [cite: 317-318]

| Core Class Mining Method | Example-F1 | P@1 | P@3 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Explicit Mention | 0.1611 | 0.2168 | 0.1564 | 0.2045 |
| 0Shot | 0.4793 | 0.7361 | 0.4782 | N/A |
| **Ours** | **0.5431** | **0.7918** | **0.5414** | **0.5911** |
| Ours-NoCS | 0.3812 | 0.6254 | 0.3831 | 0.4366 |
| Ours-NoConf | 0.2603 | 0.4431 | 0.2521 | 0.3014 |

Table 3 lists all the results. [cite_start]First, we find that the "Explicit Mention" method, which treats all classes with names explicitly appear in the corpus as the core classes, does not perform well for our HMTC problem. [cite: 359] [cite_start]One reason could be many class names are human-curated summarizing phrases that do not appear in the corpus naturally. [cite: 360] [cite_start]Second, the "0Shot" method views the output classes of baseline method Hier-0Shot-TC as the core classes and trains a new classifier. [cite: 361] [cite_start]Interestingly, this new classifier performs better than the original Hier-0Shot-TC classifier, which shows that transferring knowledge from a general zero-shot classifier to a domain-specific classifier is a possible and promising direction. [cite: 362] Finally, we compare variants of our own methods. [cite_start]The "Ours-NoCS" method removes the candidate core class selection step (c.f. Sect. 3.2.1) and treats all classes with high confidence scores as core classes. [cite: 363] [cite_start]The "Ours-NoConf" method skips the confident core class identification step (c.f. Sect. 3.2.2) and views all candidate core classes as the final output core classes. [cite: 364] [cite_start]We can see a significant performance drop on both ablations, which shows the importance of our two core class mining steps. [cite: 365]

### 4.7 Analysis of Classifier Architecture
[cite_start]We study whether we can use the identified document core classes to train other text classifiers with different architectures such as fastText (Joulin et al., 2016) and TextCNN (Kim, 2014). [cite: 366] [cite_start]As shown in Table 4, both methods achieve reasonable performance. [cite: 367] [cite_start]We can also see that TaxoClass with and without GNN-enhanced class encoder can outperform both methods. [cite: 368] [cite_start]This shows the effectiveness of our dual-encoder style classifier architecture. [cite: 369]

[cite_start]**Table 4: Performance of different classifiers on Amazon-531 dataset.** [cite: 357]

| Method | Example-F1 | P@1 | P@3 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| fastText | 0.4472 | 0.7515 | 0.4521 | 0.4587 |
| TextCNN | 0.4787 | 0.7694 | 0.4771 | 0.4827 |
| TaxoClass-NoGNN | 0.5271 | 0.7642 | 0.5213 | 0.5621 |
| **TaxoClass** | **0.5934** | **0.8120** | **0.5894** | **0.6332** |

### 4.8 Supervision Signals in Class Names
[cite_start]We vary the percentage of labeled documents on Amazon-531 dataset for training a supervised fastText classifier and present its corresponding performance in Fig. 5. We can see the performance of our TaxoClass framework is equivalent to that of supervised fastText learned on roughly 70% of labeled documents in the training set (i.e., about 20,000 labeled documents). [cite: 392, 378]

## 5. Related Work
[cite_start]**Weakly-supervised Text Classification.** There exist some previous studies that leverage a few labeled documents or class-indicative keywords as weak supervision signals for text classification. [cite: 380] [cite_start]A pioneering method is dataless classification (Chang et al., 2008; Song and Roth, 2014) which embeds documents and classes into the same semantic space of Wikipedia concepts and performs classification using the embedding similarity. [cite: 381] Li et al. (2018, 2019) [cite_start]extend this idea by mining concepts directly from the corpus rather than using the external Wikipedia. [cite: 382] Along another line, Chen et al. (2015) and Li et al. (2016) [cite_start]propose to apply a seed-guided topic model to infer class-specific topics from class-indicative keywords and to predict document classes from posterior class-topic assignments. [cite: 383-384] [cite_start]Compared with these methods, our TaxoClass framework neither restricts document and class embeddings to live in the same semantic space nor imposes strong statistical assumptions. [cite: 385]

Recently, neural models are applied to weakly-supervised text classification. Meng et al. (2018, 2019) [cite_start]propose a pretrain-and-refine paradigm which first generates pseudo documents to pretrain a neural classifier and then refine this classifier via self-training. [cite: 386-387] Mekala and Shang (2020); Meng et al. (2020); Wang et al. (2020) [cite_start]improve the above methods by introducing contextualized weak supervision and using a pre-trained language model to obtain better text representations. [cite: 388-389] [cite_start]While achieving inspiring performance, these methods all assume each document has only one class and all class names (or class-indicative keywords) must appear in the corpus for pseudo training data generation. [cite: 390] [cite_start]In this paper, we relax these assumptions and develop a new method for weakly-supervised hierarchical multi-label text classification task. [cite: 391, 394]

[cite_start]**Zero-shot Text Classification.** Zero-shot text classification learns a text classifier based on training documents belonging to seen classes and applies the learned classifier to predict testing documents belonging to unseen classes (Wang et al., 2019). [cite: 395] Nam et al. (2016) [cite_start]jointly embed documents and classes into a shared semantic space where knowledge from seen classes can be transferred to unseen classes. [cite: 396] [cite_start]Such an idea is further developed in (Rios and Kavuluru, 2018; Srivastava et al., 2018; Yin et al., 2019; Chu et al., 2020) where external resources (e.g., knowledge graphs, natural language explanations of unseen classes, and open domain data) are introduced to help learn a better shared semantic space. [cite: 397] [cite_start]Comparing with these methods, our TaxoClass framework does not require labeled data for a set of seen classes. [cite: 398]

[cite_start]**Hierarchical Text Classification.** Hierarchical text classification leverages a class hierarchy to improve the standard text classification performance. [cite: 399] [cite_start]Typical methods can be divided into two categories: (1) local approaches which learn a text classifier per class (Banerjee et al., 2019), per parent class (Liu et al., 2005), or per level (Wehrmann et al., 2018), and (2) global approaches which incorporate taxonomy structure information into one single classifier through recursive regularization (Gopal and Yang, 2013) or graph neural network (GNN) based encoder (Peng et al., 2018; Huang et al., 2019; Zhou et al., 2020). [cite: 400] [cite_start]Our TaxoClass framework adopts the second global approach and uses a GNN-based encoder to obtain each class's representation. [cite: 401]

## 6. Conclusions & Future Work
[cite_start]This paper studies the hierarchical multi-label text classification problem when only class surface names, instead of massive labeled documents, are given. [cite: 403] [cite_start]We propose a novel TaxoClass framework which leverages the class taxonomy structure to derive document core classes and learns taxonomy-enhanced text classifier for prediction. [cite: 404] [cite_start]Extensive experiments demonstrate the effectiveness of TaxoClass on two real-world datasets from different domains. [cite: 405] [cite_start]In the future, we plan to explore how TaxoClass framework can be integrated with semi-supervised methods and data augmentation methods, when some class surface names are too ambiguous to indicate class semantics. [cite: 406] [cite_start]Moreover, we consider extending our multi-label self-training method to other related NLP tasks such as fine-grained entity typing. [cite: 407-408]

**Discussion of Ethics**
[cite_start]As text classification is a standard task in NLP, we do not see any significant ethical concerns. [cite: 410] [cite_start]The expected usage of our work is to classify documents such as news articles, scientific literature, and etc. [cite: 411]

**Acknowledgements**
[cite_start]Research was sponsored in part by US DARPA SocialSim Program No. W911NF-17-C0099, NSF IIS 16-18481, IIS 17-04532, and IIS 17-41317, and DTRA HDTRA11810026. [cite: 413] [cite_start]Any opinions, findings or recommendations expressed herein are those of the authors and should not be interpreted as necessarily representing the views, either expressed or implied, of DARPA or the U.S. Government. [cite: 414] [cite_start]We thank anonymous reviewers for valuable and insightful feedback. [cite: 415]

## References
[cite_start][cite: 417] S. Banerjee, Cem Akkaya, Francisco Perez-Sorrosal, and K. Tsioutsiouliklis. 2019. Hierarchical transfer learning for multi-label text classification. In ACL.
[cite_start][cite: 418] David Berthelot, Nicholas Carlini, I. Goodfellow, Nicolas Papernot, A. Oliver, and Colin Raffel. 2019. Mixmatch: A holistic approach to semi-supervised learning. ArXiv, abs/1905.02249.
[cite_start][cite: 420] Ming-Wei Chang, Lev-Arie Ratinov, Dan Roth, and Vivek Srikumar. 2008. Importance of semantic representation: Dataless classification. In AAAI.
[cite_start][cite: 421] Wei-Cheng Chang, Hsiang-Fu Yu, Kai Zhong, Yiming Yang, and Inderjit S. Dhillon. 2019. X-bert: extreme multi-label text classification with bert. In arXiv.
[cite_start][cite: 423] Xingyuan Chen, Yunqing Xia, Peng Jin, and John A. Carroll. 2015. Dataless text classification with descriptive Ida. In AAAI.
[cite_start][cite: 424] Zewei Chu, K. Stratos, and Kevin Gimpel. 2020. Natcat: Weakly supervised text classification with naturally annotated datasets. ArXiv, abs/2009.14335.
[cite_start][cite: 425] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT.
[cite_start][cite: 427] S. Gopal and Yiming Yang, 2013. Recursive regularization for large-scale classification with hierarchical and graphical dependencies. In KDD.
[cite_start][cite: 428] Sylvain Goumy and Mohamed-Amine Mejri. 2018. Ecommerce product title classification. In SIGIR.
[cite_start][cite: 430] Chuan Fei Guo, Alireza Mousavi, Xiang Wu, Daniel N. Holtmann-Rice, Satyen Kale, Sashank J. Reddi, and Sanjiv Kumar. 2019. Breaking the glass ceiling for embedding-based classifiers for large output spaces. In NeurIPS.
[cite_start][cite: 432] Suchin Gururangan, Tam Dang, Dallas Card, and Noah A. Smith. 2019. Variational pretraining for semi-supervised text classification. In ACL.
[cite_start][cite: 433] Wei Huang, Enhong Chen, Qi Liu, Yuying Chen, Zai Huang, Yang Liu, Zhou Zhao, Dan Zhang, and Shijin Wang. 2019. Hierarchical multi-label text classification: An attention-based recurrent network approach. In CIKM.
[cite_start][cite: 435] Himanshu Jain, Yashoteja Prabhu, and Manik Varma. 2016. Extreme multi-label loss functions for recommendation, tagging, ranking, and other missing label applications. In KDD.
[cite_start][cite: 437] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. 2016. Bag of tricks for efficient text classification. arXiv preprint arXiv: 1607.01759.
[cite_start][cite: 439] Yoon Kim. 2014. Convolutional neural networks for sentence classification. In EMNLP.
[cite_start][cite: 440] Thomas Kipf and M. Welling. 2017. Semi-supervised classification with graph convolutional networks. In ICLR.
[cite_start][cite: 441] Quoc V. Le and Tomas Mikolov. 2014. Distributed representations of sentences and documents. In ICML.
[cite_start][cite: 442] Jens Lehmann, Robert Isele, Max Jakob, A. Jentzsch, D. Kontokostas, Pablo N. Mendes, S. Hellmann, M. Morsey, Patrick van Kleef, S. Auer, and C. Bizer. 2015. Dbpedia - a large-scale, multilingual knowledge base extracted from wikipedia. Semantic Web, 6:167-195.
[cite_start][cite: 444] Chenliang Li, Jian Xing, Aixin Sun, and Zongyang Ma. 2016. Effective document labeling with very few seed words: A topic model approach. In CIKM.
[cite_start][cite: 446] Keqian Li, Shiyang Li, Semih Yavuz, Hanwen Zha, Yu Su, and Xifeng Yan. 2019. Hiercon: Hierarchical organization of technical documents based on concepts. In ICDM.
[cite_start][cite: 448] Keqian Li, Hanwen Zha, Yu Su, and Xifeng Yan. 2018. Unsupervised neural categorization for scientific publications. In SDM.
[cite_start][cite: 449] T. Liu, Yiming Yang, H. Wan, H. Zeng, Z. Chen, and W. Ma. 2005. Support vector machines classification with a very large-scale taxonomy. SIGKDD, 7:36-43.
[cite_start][cite: 451] Julian J. McAuley and Jure Leskovec. 2013. Hidden factors and hidden topics: understanding rating dimensions with review text. In RecSys.
[cite_start][cite: 453] Dheeraj Mekala and Jingbo Shang. 2020. Contextualized weak supervision for text classification. In ACL.
[cite_start][cite: 454] Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2018. Weakly-supervised neural text classification. In CIKM.
[cite_start][cite: 455] Yu Meng, Jiaming Shen, Chao Zhang, and Jiawei Han. 2019. Weakly-supervised hierarchical text classification. In AAAI.
[cite_start][cite: 456] Yu Meng, Yunyi Zhang, Jiaxin Huang, Chenyan Xiong, Heng Ji, Chao Zhang, and Jiawei Han. 2020. Text classification using label names only: A language model self-training approach. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing.
[cite_start][cite: 459] Jinseok Nam, Eneldo Loza Mencía, and Johannes Fürnkranz. 2016. All-in text: Learning document, label, and word representations jointly. In AAAI.
[cite_start][cite: 460] Ioannis Partalas, A. Kosmopoulos, Nicolas Baskiotis, T. Artières, G. Paliouras, Éric Gaussier, Ion Androutsopoulos, M. Amini, and P. Gallinari. 2015. Lshtc: A benchmark for large-scale text classification. ArXiv, abs/1503.08581.
[cite_start][cite: 462] Hao Peng, Jianxin Li, Y. He, Yaopeng Liu, Mengjiao Bao, L. Wang, Y. Song, and Qiang Yang. 2018. Large-scale hierarchical text classification with recursively regularized deep graph-cnn. Proceedings of the 2018 World Wide Web Conference.
[cite_start][cite: 464] Yashoteja Prabhu, Anil Kag, Shrutendra Harsola, Rahul Agrawal, and Manik Varma. 2018. Parabel: Partitioned label trees for extreme classification with application to dynamic search advertising. In WWW.
[cite_start][cite: 466] Anthony Rios and Ramakanth Kavuluru. 2018. Fewshot and zero-shot multi-label learning for structured label spaces. In EMNLP.
[cite_start][cite: 467] J. Shen, Zhihong Shen, Chenyan Xiong, Chunxin Wang, Kuansan Wang, and Jiawei Han. 2020. Taxoexpan: Self-supervised taxonomy expansion with position-enhanced graph neural network. Proceedings of The Web Conference 2020.
[cite_start][cite: 469] Yangqiu Song and Dan Roth. 2014. On dataless hierarchical text classification. In AAAI.
[cite_start][cite: 470] Shashank Srivastava, Igor Labutov, and Tom M. Mitchell. 2018. Zero-shot learning of classifiers from natural language quantification. In ACL.
[cite_start][cite: 471] Wei Wang, Vincent Wenchen Zheng, Han Yu, and Chunyan Miao. 2019. A survey of zero-shot learning: Settings, methods, and applications. In ACM TIST.
[cite_start][cite: 473] Zihan Wang, Dheeraj Mekala, and Jingbo Shang. 2020. X-class: Text classification with extremely weak supervision. ArXiv, abs/2010.12794.
[cite_start][cite: 474] Jonatas Wehrmann, R. Cerri, and Rodrigo C. Barros. 2018. Hierarchical multi-label classification networks. In ICML.
[cite_start][cite: 476] Huiru Xiao, Xin Liu, and Y. Song. 2019. Efficient path prediction for semi-supervised and weakly supervised hierarchical text classification. The World Wide Web Conference.
[cite_start][cite: 478] Junyuan Xie, Ross B. Girshick, and Ali Farhadi. 2016. Unsupervised deep embedding for clustering analysis. In ICML.
[cite_start][cite: 479] Peng Xu and Denilson Barbosa. 2018. Neural fine-grained entity type classification with hierarchy-aware loss. In The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL 2018).
[cite_start][cite: 482] Wenpeng Yin, Jamaal Hay, and Dan Roth. 2019. Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. In EMNLP/IJCNLP.
[cite_start][cite: 484] Ronghui You, Suyang Dai, Zihan Zhang, Hiroshi Mamitsuka, and Shanfeng Zhu. 2019. Attentionxml: Extreme multi-label text classification with multi-label attention based recurrent neural networks. In NeurIPS.
[cite_start][cite: 486] Yue Yu, Simiao Zuo, Haoming Jiang, W. Ren, Tuo Zhao, and C. Zhang. 2020. Fine-tuning pre-trained language model with weak supervision: A contrastive-regularized self-training approach. ArXiv, abs/2010.07835.
[cite_start][cite: 488] Ziqian Zeng, Wenxuan Zhou, Xin Liu, and Yangqiu Song. 2019. A variational approach to weakly supervised document-level multi-aspect sentiment classification. In NAACL-HLT.
[cite_start][cite: 490] Jie Zhou, Chunping Ma, Dingkun Long, Guangwei Xu, Ning Ding, Haoyu Zhang, Pengjun Xie, and G. Liu. 2020. Hierarchy-aware global model for hierarchical text classification. In ACL.