# Leveraging Relations

지금까지는 우리는 각각의 `entity(text)` 들을 encoder(BERT)를 이용해 각각 vector로 만듦 ⇒ 👎 하지만 이것은 entity 사이에 관계를 무시한다는 문제점이 존재 (category와 제품간의 관계)

- 각 entity는 다른 entity혹은 category와의 관계를 고려함으로써 완벽히 이해가 될 수 있음
1. `Node Representation` : 이러한 relation을 고려한 vector를 어떻게 학습할 수 있을까?
2. `GNN` : 어떻게 graph-structured data를 딥러닝에 사용이 가능할까?

# Node Representation

우리가 맨 처음 해야하는 것: entity ⇒ vector로 표현 

- 어떻게 각각의 node들을 **저차원 공간에서 vector로 표현**할 수 있는지 알아보고자 함

💡 목표: **실제 graph에서 유사한 node**끼리 **embedding space에서 유사하도록 encode 하는 것**

1. `encoder`: 어떻게 해당 그래프를 encode 시킬 것인가!
    - **Skip-Gram learning**과 유사하게 학습 ⇒ embedding table을 lookup하는 방식 사용 (무작위로 초기화 된 뒤, 의미있는 의미를 가지도록 학습)
2. `similarity function` : 어떻게 기존 그래프에서의 유사도를 측정해, vector space에서도 유사한지 볼 것인가?
    - 어떤 **node들이 유사하다고 볼 수 있을까?** (직접 연결된 것? ,같은 노드를 공유하는 것?, 비슷한 역할을 하는 것?)

![image.png](Leveraging%20Relations/image.png)

<aside>
📖

Skip Gram learning (복습)

1. 각 단어를 embedding table을 이용하여 vector space에서 mapping을 진행 (해당 embedding들은 무작위로 초기화되어 있음)
2. 비슷한 문맥 (context)에 단어가 같이 존재할 때 word embedding은 점점 유사해짐(높은 내적값)
    - `distributional hypothesis` : 비슷한 context에서 자주 등장하는 단어들은 **비슷한 의미를 가질 것이다라는 가정에서 출발!!**

![image.png](Leveraging%20Relations/image%201.png)

</aside>

## Similarity: Random Walk node embedding

[arxiv.org](https://arxiv.org/pdf/1403.6652)

해당 알고리즘은 다음과 같은 방법을 이용한다. 

1. 그래프와 시작 node가 주어졌을 때, 이어져 있는 노드(neighbors) 중 하나를 무작위로 선택한 후 이동
2. 해당 node에서 또 다시 무작위로 neighbors를 선택해 이동

방문한 node들의 sequence가 곧 그래프의 `random walk` 를 의미 ⇒ **각 node가 `단어` 라고 했을 때 random walk의 결과는 `문장` 이 된다고 볼 수 있음** 

즉, 자연어에서 random walk가 곧 단어들의 집합체인 문장이 된다는 것을 의미하며, 이를 통해 유사도를 정의할 수 있으며, `skip gram` 를 이용해 학습이 가능해진다!

- **2개의 인접한 노드**: **`local` 한 유사도**를 의미 ⇒ `random walk` 에서도 자주 등장할 가능성 높음
- **2개의 먼 노드**: **`global` 한 유사도**를 의미 ⇒ `random walk` 에서 두 노드 사이를 잇는 **여러 길**들이 존재할 경우 같이 자주 등장할 가능성 높음

1. `Word2vec` 과 같이 `center node(w)` 를 이용해 `context node(c)` 를 예측하는 확률을 최대화하도록 최적화 학습을 진행 (window size도 존재)
    - 즉, 주어진 중심 단어로부터 주변 단어를 예측하도록 만들어 자주 등장하는 단어들은 vector space에서 가깝게 만든다

![image.png](Leveraging%20Relations/image%202.png)

1. 이럴 때 log probability가 곧 두 embedding간의 내적을 의미하게 됨 (softmax에 log를 취하면 지수값이 나오는데 이때 이 값이  내적값과 비례- 나머지 constant)
    - positive pair: 중심(center) 단어 근처의 주변(context) 단어간의 쌍
    - negative pair: 중심단어 sliding window안에 없는 모든 단어와의 쌍

![image.png](Leveraging%20Relations/image%203.png)

1. 위와 같이 모든 단어 쌍에 대해 확률 표현 가능 ⇒ `softmax` 를 통해 확률 분포를 만들 수 있음

![image.png](Leveraging%20Relations/image%204.png)

😭 문제점: 위의 식을 보게 되면 모든 vocabulary V에 대해서 분모 계산을 진행 ⇒ 이는 backpropagation 진행시 매우 오랜 시간이 걸리게 됨 

1. `negative sampling` : 몇몇 negative sample에 대해서만 내적 연산을 진행해 update 진행
    - ⭐️식에서 볼 수 있듯, **log를 취하게 되면 두 백터의 내적**이 나오게 됨
    - ⭐️ 더 이상 softmax로 확률을 계산하지 않고 진짜 문맥쌍인지 맞추는 `binary classification` 문제로 전환함 ⇒ 😎이로 인해 sampling을 하여 분모가 많이 작아지고 분자에 영향이 커져 제대로 학습을 못하게 되지 않음

![image.png](Leveraging%20Relations/image%205.png)

<aside>
📖

Negative Sampling ⇒ Binary classification

`Negative Sampling`을 통해 더 이상 softmax가 아닌 **binary classification 형태인 sigmoid**로 실제 문맥쌍인지를 **** 학습

- **softamx로 주변단어(context word) 분포를 모델링**하지 않고 **sigmoid로 실제 문맥쌍인지를 모델링**
- 이렇게 근사적인 objective function을 만드는 것이 최적화에 용이

![image.png](Leveraging%20Relations/image%206.png)

중심단어(w)와 주변단어(c)의 관계가 False일 때는 1-objective function인데 이것이 곧 **sigmoid 함수의 성질에 따라, 두 단어의 내적값이 작을 수록 커지게** 되므로 우리가 학습하고자하는 목표에 알맞게 됨

![image.png](Leveraging%20Relations/image%207.png)

**Loss function이 다음과 같이 변하게 됨**

</aside>

이렇게 random walk 방식으로 similarity를 정의하여 저차원 공간에 embedding을 학습을 시킬 수 있으며 이를 시각화하면 다음과 같은 결과가 나오게 된다

![image.png](Leveraging%20Relations/image%208.png)

## Similarity: Node2vec

[arxiv.org](https://arxiv.org/pdf/1607.00653)

그렇다면 조금 더 잘 `walk` 할 수 있지 않을까? node embedding이 학습하는 것은 결국에 **우리가 graph에서 어떻게 걸었느냐에 의존**하기 때문에 random walk의 문제점이 있을 경우 해당 문제점을 그대로 학습할 수 있음

👎 **높은 `degree` 의 노드들을 너무 자주 지나며, 덜 연결된 영역을 간과함**

- `degree` : 각각의 노드에 연결된 edge의 수

💡 완전히 무작위하게 walk하는 것이 아닌, **유연하게 walk 전략을 조절하자!**

⭐️ 그래프의 `local` 과 `global`  관점의 balance를 취하자 ⇒ **둘 중 하나의 관점으로 편향되지 말자!(random walk는 local view에 biased)**

- `local view` : 근처에 가까이 연결된 node들에 집중
- `global view` : 더 멀리, 많이 연결된 (multi-hop) node를 찾아 가자

그렇다면 어떻게 두 관점 사이에서 balance를 할 수가 있을까? ⇒ **두가지 방식의 search 전략을** 사용

1. BFS-style: `local neighborhoods` 를 포착
2. DFS-style: `global structure` 을 포착

그렇다면 두 search 전략을 **어떻게 섞어서(interpolate) 사용**할까?

🅰️ edge에 (unnormalized) `transition probability` 를 명시하여 random walk를 진행 

![image.png](Leveraging%20Relations/image%209.png)

1. s1으로 돌아감: **BFS도 결국 다시 이전 노드로 돌아간 뒤, depth+1에 있는 다른 node를 탐색하는 것**이기 때문에 s1으로 돌아가는 것이 BFS!
2. s2로 이동: `neutral` ⇒ s1을 기준으로 같은 거리에 있음 (중립적으로 탐색 진행)
3. s3로 이동: s1를 기준으로 depth가 증가했기 떄문에 DFS라고 볼 수 있음

⇒ parameter `p,q` 의 가중치에 따라 BFS로 갈지 DFS로 갈지 정할 수 있음 

- `q<1` : 더 멀리 갈 확률이 증가하여 DFS (`p>1`)
- `q>1` : 멀리 갈 확률이 적어지므로 BFS (`p<1`)

![image.png](Leveraging%20Relations/image%2010.png)

Node2vec 알고리즘

1. 그래프가 주어지고 edge(t,w)에 대해 w에서 시작하는 `edge transition probability` 를 계산
2. **미리 계산된 `transition probability` 를 기반으로 random walk simulation 진행**
3. 이렇게 지나간 경로(문장)을 이용해 `skip-gram learning` 진행

이렇게 만들어진 node embedding은 node classification(해당 embedding을 feature로 사용) 혹은 link prediction에 사용(비슷한 노드들이 들어왔을 때 빠진 edge를 예측)

# GNN

무항향 그래프 G가 있고 인접행렬 A가 존재할 때 deep learning 입력으로 인접행렬 A와 해당 node의 feature 행렬(나이, 성별 등..) X을 합친 행렬을 넣을 수 있음

![image.png](Leveraging%20Relations/image%2011.png)

👎 **문제점**

1. `Fixed input size` : node의 개수가 달라질 경우, input의 크기가 달라져 해당 그래프를 모델에 넣을 수 없음
2. `Permutation sensitivity` : 인접행렬의 node 순서는에 따라 model의 결과가 다르게 나옴
    - 모델에 들어가는 node의 순서가 달라져도 graph가 똑같기 때문에 결과는 동일하게 나와야함
    - 단순한 `FC layer` 는 weight가 고정되어있기 때문에 입력 순서가 달라지면 결과가 달라짐
    - 😎 `CNN` 에서도 결국 fc layer과 다른 `translation-equivariance` 성질 때문에 효과적 ⇒ ⭐️즉, **CNN은 `이미지` 에 특화된 성질을 기반으로 모델을 만들었다면, GNN도 `그래프` 에 특화된 성질을 기반으로 모델을 만들어야함**
- 그래프는 `canonical ordering` (노드의 순서를 일정하게 정하는 규칙) 을 가지고 있지 않음 ⇒ node가 어떻게 입력이 되는지 output은 같아야함

![image.png](Leveraging%20Relations/image%2012.png)

👎 CNN과 MLP와 같은 기존 모델 구조는 `permutation equivariant` 하지 않음

- 😎 CNN: 이미지에 잘 작동하는 이유는 **공간 구조가 일정 (negihborhood에 동일한 kernel이 적용)** ⇒ 결국 kernel(weight)는 고정된 공간 위치에 연결 ⇒ (1,1) pixel과 (10,10) pixel이 바뀌면 CNN의 결과가 완전히 달라짐
- 즉 CNN에서 convolution filter도 **위치 기반으로 동작**

## Propagation information over graph (label, feature)

**가정** : node 분류 문제 & 일부의 노드만이 label ⇒ label이 없는 노드들이 많은 상황

즉, `semi-supervised learning` 상황이지만, 여기에 **우리는 node간의 관계를 알고 있음** (new!!)

⇒ node 간의 관계를 알고 있는 것을 통해 `label info` 를 propagate 시킬 수 있음

⇒ iterative하게 자신을 포함한 neighboring node을 평균 내어 node의 label을 update 진행

⇒ 수렴할 때까지 지속

![image.png](Leveraging%20Relations/image%2013.png)

![image.png](Leveraging%20Relations/image%2014.png)

이러한 `propagated label` 들을 unlabeled node에 대한 pseudo-label로 사용이 가능 ⇒ 이런 식으로 그래프 상에서 정보를 propagate함으로써, **그래프 내의 context에서 각각의 entity들을 더욱 이해할 수 있게 됨**

🤔 그렇다면 label대신에 node의 feature을 propagate해도 되지 않을까?

- classifier는 간접적으로 unlabeled node로부터 학습을 할 수 있게 됨 (각각의 node feature들이 주변의 neighbor nodes에 의해 정보를 갖게 됨)
- ⭐️ label propagation은 (label 존재 node) → (label 존재 X node)로의 단방향 과정이지만 `feature propagation` 은 **node간의 message passing**이 되기 떄문에, **unlabeled node의 feature가 labeled node feature에도 섞이게 됨**
    - 🙋‍♀️ 조금 이해가 안가네..

![image.png](Leveraging%20Relations/image%2015.png)

최근에 들어서 2개의 방식이 모두 equivalent하다는 것이 증명됨

1. label propagate ⇒ unlabeled data를 위한 pseudo-label들을 이용해 classifier 학습
2. feature propagate ⇒ labeled data를 가지고 학습 (labeled data feature에 이미 unlabeled data feature가 들어있는 상황)

`feature propagation` 은 **1) limited label**과 **2) relational info**를 모두 활용하기에 효과적인 방법 

🤔 그렇다면 해당 propagation을 deep learning을 이용해 더 강력하게 만들 수 있지 않을까?

## GCN (Graph Convolutional Networks)

[arxiv.org](https://arxiv.org/pdf/1609.02907)

GNN에 수많은 모델 중 하나

`key idea` : local network neighborhoods를 기반으로 node embedding을 생성 (이때 각 node는 **neural network를 이용해 neighbors의 정보를 합침**)

- 각 node는 자신만의 computation graph를 가지게 됨 (neighborhood 구조에 따라)

![image.png](Leveraging%20Relations/image%2016.png)

- Layer-0: node u의 embedding이 자기자신의 input feature인 $x_u$
- Layer-K: 해당 embedding은 K hop이 떨어진 노드들로부터의 정보를 모은 것으로 표현이 됨

결국 각각의 GNN layer에서 하고자 하는 것은

1. **neighbor information을 모음 (input을 만듦)**
2. **모은 정보를 neural network를 통해 변환 (input을 모델에 통과)**

### Forward pass

**K-layer GCN forward pass**

![image.png](Leveraging%20Relations/image%2017.png)

- $1/|N(v)|$를 통해 각각의 이전 layer embedding 값이 일정하게 평균적인 영향을 미치도록 만듦
- $W_k$ 이외에 따로 $B_k$가 존재하는 것은 **neighbor에서 얻은 정보**와 **자기자신의 정보**를 구분해서 처리 ⇒ **서로 다른 가중치를 부여**해 자신의 정보를 유지하는 역할을 진행
    - 😎다만, 더 간단한 형태로 두 matrix를 합쳐서 사용하기도 한다
    
    ![image.png](Leveraging%20Relations/image%2018.png)
    

![image.png](Leveraging%20Relations/image%2019.png)

- 최종적인 embedding은 **1)** node 자체의 feature (by matrix B)와 **2)** neighbor info (by matrix W) 모두의 정보를 가짐 ⇒ downstream task에 사용이 가능

![image.png](Leveraging%20Relations/image%2020.png)

- 각 layer마다 모든 node에 걸쳐 같은 aggregation parameter가 사용이 됨

🤔 왜 이전 layer에서의 노드가 다름에도 같은 parameter을 공유할까?

`🅰️ permutation equivariant` : a모든 node에서 같은 변환 규칙을 적용하여 같은 결과를 만들어냄 (단순히 값을 더함으로 node의 위치가 중요하지 않음) ⇒ 다른 parameter 사용하면 node의 순서가 중요해짐

Optimization: 최종 embedding에 대해서 binary classification을 진행하여 W,B 등의 parameter에 대한 학습을 진행

![image.png](Leveraging%20Relations/image%2021.png)

### Matrix formulation

각각의 GCN layer들은 matrix multiplication을 통해 효과적으로 연산이 가능 ⇒ **GPU에서 parallel하게 연산을 진행하기 위해!**

1. node embedding을 matrix 형태로 만듦 
- $\mathbf{H}^{(k)}=\begin{bmatrix}(\mathbf{h}_1^{(k)})^\top\\ \vdots\\ (\mathbf{h}_{|V|}^{(k)})^\top\end{bmatrix}$
1. neighbors aggregation을 인접행렬 A를 이용해 만듦
    - $\sum_{u \in N(v)} \mathbf{h}_u^{(k)} = \mathbf{A}_{v,:} \mathbf{H}^{(k)}$
    - 이때 $A_{v,:}$ 는 row vector을 의미하며 서로 이웃(neighbor)인 node일 때는 1의 값을, 아닐 떄는 0인 값을 가지고 있어 neighbors 표현 가능 ⇒ **Matrix multiplication을 통해 인접한 이웃만을 표현 가능**
    - ex) v1이 v2,v3,v4와 연결이 되었을 때 
    
    $\mathbf{A}_{v_1,:} \mathbf{H}^{(k)} = \mathbf{h}_{v_2}^{(k)} + \mathbf{h}_{v_3}^{(k)} + \mathbf{h}_{v_4}^{(k)}$
2. diagonal degree matrix 구성 ⇒ 이전 노드의 영향을 평균적으로 주기 위해 미리 degree를 게싼
    - $\mathbf{D}_{v,v} = \mathrm{Deg}(v) = |N(v)|$ , 
    
    $\mathbf{D} \in \mathbb{R}^{|V| \times |V|}$
    - $\mathbf{D}^{-1}_{v,v} = \frac{1}{|N(v)|}$ : inverse도 역시 diagonal

![image.png](Leveraging%20Relations/image%2022.png)

1. neighbor aggregation이 다음과 같이 변환
    - $\sum_{u \in N(v)} \frac{\mathbf{h}_u^{(k-1)}}{|N(v)|}
    \;\;\Rightarrow\;\;
    \mathbf{H}^{(k+1)} = \mathbf{D}^{-1} \mathbf{A} \mathbf{H}^{(k)}$
    - $D^{-1}$ ⇒ 모든 node에서 $1/{N(v)}$ 의 값을 가지고 있음
    - $AH^{(k)}$ ⇒ 인접한 노드에서의 embedding 값 aggregate (sum)
2. 각 GCN layer을 matrix 형태로 다시 쓸 수 있음

![image.png](Leveraging%20Relations/image%2023.png)

<aside>
📖

summary

1. 각 node는 neural network를 통해 neighbors 정보를 aggregate
2. input node X는 text, image등을 포함
3. Multi-layer transformation을 통해 GNN은 점진적으로 고차원 feature을 학습
</aside>

<aside>
📖

코딩!

⭐️ 결국 GNN의 목표는 **데이터를 변환(MLP,CNN:feature 추출하는 변환)하는 것이 아닌 node(특정 embedding)의 의미를 더 정교하고 풍부하게 만들어 classification이나 search와 같은 downstream task에 도움을 줌 ⇒ 보정 filter!**

1. 각 **node**에 해당하는 논문**들(논문의 단어)**을 BOW로 embedding → 해당 embedding으로 **X**를 생성한 뒤 어떤 **label**에 해당하는지를 학습
2. 
    - x: 각 node에 해당하는 text들(**상품** 설명)을 BERT로 embedding
    - E: 각 node는 해당하는 text들(category 이름)을 BERT로 embedding→ 해당 embedding으로 E를 생성한 뒤에, 학습을 통해 **단순한 단어 뜻이었던 label vector**에서 **계층구조 문맥이 포함된 label vector**로 변환
    - 둘의 내적을 통해 상품 vector x와 가장 유사한 구조적으로 강화된 label vector E를 찾게 됨
    - 코드
        
        ```python
        # ==========================================================
        # Your Task: Implement Label GCN and GCN-Enhanced Classifier
        # ==========================================================
        
        class LabelGCN(nn.Module):
            """
            Multi-layer Graph Convolutional Network (GCN) encoder for label embeddings.
        
            Each layer should perform the following steps:
                1. Aggregate neighbor embeddings: H <- A_hat @ H
                2. Linear transformation: H <- H @ W
                3. (Optional) Apply ReLU and Dropout (skip for the last layer)
        
            Args:
                emb_dim (int): Dimension of label embeddings.
                num_layers (int): Number of GCN layers.
                dropout (float): Dropout probability.
            """
            def __init__(self, emb_dim, num_layers=1, dropout=0.5):
                super().__init__()
                # TODO: Define learnable weight matrices (list of emb_dim x emb_dim parameters)
                # Hint: Use nn.ParameterList and Xavier uniform initialization
                self.weights=nn.ParameterList(
                    [nn.Parameter(torch.empty(emb_dim,emb_dim))
                      for i in range(num_layers)
                    ]
                )
                self.num_layers=num_layers
                self.dropout=dropout
                for i in range(num_layers):
                  nn.init.xavier_uniform_(self.weights[i])
                
        
            def forward(self, H, A_hat):
                """
                Args:
                    H (torch.Tensor): Initial label embeddings, shape (num_labels, emb_dim).
                    A_hat (torch.Tensor): Normalized adjacency matrix, shape (num_labels, num_labels).
        
                Returns:
                    torch.Tensor: Updated label embeddings, shape (num_labels, emb_dim).
                """
                # TODO: Implement multi-layer GCN
                # for each layer:
                #   1) propagate messages: H = A_hat @ H
                #   2) linear transform: H = H @ W
                #   3) if not last layer: apply ReLU + Dropout
               
                for i in range(self.num_layers-1):
                  H=torch.matmul(A_hat,H) # 각 layer마다 A와 연산을 해줘야함
                  H=H@self.weights[i]
                  H=torch.nn.functional.relu(H)
                  H=torch.nn.functional.dropout(H,self.dropout)
        
                H=torch.matmul(A_hat,H)
                H=H@self.weights[self.num_layers-1]
        
                return H
        
        class GCNEnhancedClassifier(nn.Module):
            """
            Classifier that combines:
              - Document representations projected into label space
              - Label embeddings refined by a GCN over the label graph
        
            Args:
                input_dim (int): Dimension of input document embeddings.
                label_init_emb (torch.Tensor): Initial label embeddings, shape (num_labels, emb_dim).
                A_hat (torch.Tensor): Normalized adjacency matrix of labels, shape (num_labels, num_labels).
                num_layers (int): Number of GCN layers.
                dropout (float): Dropout probability.
            """
            def __init__(self, input_dim, label_init_emb, A_hat, num_layers=1, dropout=0.5):
                super().__init__()
                # TODO:
                # 1. Define projection layer (input_dim -> emb_dim)
                # 2. Define GCN encoder for label embeddings
                # 3. Make label_init_emb trainable (nn.Parameter)
                # 4. Register adjacency matrix (use register_buffer)
                self.W_projection=nn.Parameter(torch.empty(input_dim,label_init_emb.shape[1]))
                nn.init.xavier_uniform_(self.W_projection)
                self.GCN_encoder=LabelGCN(label_init_emb.shape[1],num_layers,dropout)
                
                # 단순히 tensor가 아닌 parameter로 만들어야함
                self.label_embeddings=nn.Parameter(label_init_emb)
        
                # parameter(학습 대상)이 아니지만, 모델과 함께 GPU로 이동해야하므로 register_buffer을 사용
                self.register_buffer('A_hat',A_hat)
        
                self.num_layers=num_layers
                self.dropout=dropout
                
        
            def forward(self, x):
                """
                Args:
                    x (torch.Tensor): Input embeddings for documents, shape (batch_size, input_dim).
        
                Returns:
                    torch.Tensor: Logits for classification, shape (batch_size, num_labels).
                """
                # TODO:
                # 1. Refine label embeddings using GCN
                # 2. Project input x into label embedding space (+ dropout)
                # 3. Compute similarity (inner product) between x_proj and label_emb
                label_embedding=self.GCN_encoder(self.label_embeddings,self.A_hat)
                
                x=x@self.W_projection
                
                x=torch.nn.functional.dropout(x,self.dropout)
                
                inner_product=x@label_embedding.T
                return inner_product
        ```
        
    

`Cora DataSet`을 사용 ⇒ 해당 paper가 어떤 분야의 paper인지 맞추는 classification task

- `feature matrix` : $X \in R^{NXd}$
    - 𝑁=2708 (number of nodes/papers),  𝑑=1433 (feature dimension)
    - Each row corresponds to one paper's bag-of-words features.
    - node embedding이 곧 bags of words feature!
- `Label vector` : $y \in \{0,...,C-1\}^N$
    - 𝐶=7 (number of research categories)
    - Each entry is the label of a paper.
- `Normalized adjacency matrix` : $\hat A \in R^{NXN}$
    - Represents graph connectivity.
    - We start with an adjacency matrix  where 𝐴=1or 0 and Then normalize it using: 𝐴̂=𝐷−12𝐴𝐷−12, where 𝐷 is the degree matrix.

![image.png](Leveraging%20Relations/image%2024.png)

![image.png](Leveraging%20Relations/image%2025.png)

맨 처음에 validation 정확도가 살짝 줄어드는건 **random한 weight로 neighbors 정보를 섞기 때문에 어떤 것이 중요한지 알 수 없음** ⇒ 초기에 성능이 안좋은 대표적인 이유

- adjacency matrix 코드
    
    ```python
    def load_cora(content_path="cora.content", cites_path="cora.cites"):
        # --- 1) Load node features & labels ---
        idx_features_labels = np.genfromtxt(content_path, dtype=str)        # Each row: [paper_id, word_attributes..., class_label]
        features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32) # Node features (1433-dim bag-of-words)
        labels_raw = idx_features_labels[:, -1]                             # Raw labels (e.g., "Neural_Networks")
    
        classes = sorted(set(labels_raw))                                   # Unique class names
        label_map = {c: i for i, c in enumerate(classes)}                   # Map class → integer
        labels = np.array([label_map[l] for l in labels_raw], dtype=np.int64) # Integer labels
    
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)           # Original paper IDs
        idx_map = {j: i for i, j in enumerate(idx)}                         # Map paper ID → row index
    
        # class와 paper_id를 integer로 만들고 feature들을 독립적인 array로 생성
    
        # --- 2) Load edges ---
        edges_unordered = np.genfromtxt(cites_path, dtype=np.int32)         # Each row: [citing_paper_id, cited_paper_id]
        edges = np.array(list(map(lambda x: [idx_map[x[0]], idx_map[x[1]]], edges_unordered)))  
    
        # 각 edge에 연결된 node들을 index(row)로 연결
        
        # --- 3) Build adjacency matrix ---
        n_nodes = labels.shape[0]                                           # Total number of nodes
        A = np.eye(n_nodes, dtype=np.float32)                               # Identity matrix (self-loops)
        for i, j in edges:
            A[i, j] = 1                                                     # Add edge i → j
            A[j, i] = 1                                                     # Add edge j → i (undirected)
    
        D = np.sum(A, axis=1)                                               # Degree matrix (node degrees)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))                       # D^(-1/2)
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt                                 # Normalized adjacency: Â = D^(-1/2) * A * D^(-1/2)
    
        return torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(A_hat).float()
      
    ```
    
- GCN layer code
    
    ```python
    # === Define GCN Layer ===
    class GCNLayer(nn.Module):
        """
        A single Graph Convolutional Network (GCN) layer.
    
        Operation:
            H' = A_hat @ H @ W
            where
                H : [N, d_in]   (input node features)
                A_hat : [N, N] (normalized adjacency with self-loops)
                W : [d_in, d_out] (learnable weights)
                H' : [N, d_out] (output node features)
        """
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.W = nn.Parameter(torch.empty(in_dim, out_dim))   # learnable weight
            nn.init.xavier_uniform_(self.W)                      # Xavier initialization
    
        def forward(self, H, A_hat):
            return torch.matmul(torch.matmul(A_hat, H), self.W)
    
    # === Define GCN Classifier (2-layer GCN + Linear head) ===
    class GCNClassifier(nn.Module):
        """
        A 2-layer GCN followed by a linear classifier for node classification.
    
        Input:
            X : [N, input_dim]   (node features)
        Output:
            logits : [N, C] (class scores for each node)
    
        Architecture:
            X → GCN(input_dim → hidden_dim) → ReLU → Dropout →
                GCN(hidden_dim → hidden_dim) → ReLU → Dropout →
                Linear(hidden_dim → C)
        """
        def __init__(self, input_dim, hidden_dim, num_classes, A_hat, dropout=0.5):
            super().__init__()
            self.gcn1 = GCNLayer(input_dim, hidden_dim)
            self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
            self.fc   = nn.Linear(hidden_dim, num_classes)
    
            self.dropout = dropout
            self.register_buffer("A_hat", A_hat)  # keep adjacency as buffer (not a parameter)
    
        def forward(self, X):
            H = F.relu(self.gcn1(X, self.A_hat))
            H = F.dropout(H, p=self.dropout, training=self.training)
    
            H = F.relu(self.gcn2(H, self.A_hat))
            H = F.dropout(H, p=self.dropout, training=self.training)
    
            logits = self.fc(H)
            return logits
    ```
    
</aside>

<aside>
😎

왜 GCN이 효과가 좋을까?

1. **Message Passing**
    - Adjacency Matrix A를 곱합으로써 해당 노드와 연결된 neighbor node의 feature을 가져와 합침 ⇒ `homophily(동질성)` : 내 정보가 불확실하더라도 **이웃정보와 smoothing을 통해 nosie 제거 되고 실제 특징이 뚜렷해짐 (like minibatch)**
    - `MLP`: 내 node feature만 사용
2. **효율적인 parameter 구성**
    - translation invariance: 모든 node에 동일한 W를 공유
    - `MLP` : W가 feature 차원에만 의존
3. **의미있는 초기 embedding 발전 가능**
    - 의미있지만 불완전한 초기 feature를 graph 구조를 이용해 보정

MLP도 모든 노드에 대해서 하나의 weight만을 사용하고 GCN도 모든 노드에 대해서 하나의 weight만 사용

⭐️근데 **GCN은 CNN과 비슷하게 모든 input이 아니라 관련이 있는 input에 대해서만 weight multiplication을 진행하니까 효과적이면서 더 높은 성능을 보임**

1. **local connectivity:** 관련있는 input과만 연결
2. **Inductive bias** : 연결되어 있는 애들과는 비슷할 거라는 `homophily` 를 가정
3. **Sparsity** : MLP 처럼 모든 노드와 연산하지 말고 관련 있는 부분만 가져옴
</aside>

- 코드2
    
    <aside>
    📖
    
    d
    
    - x: 각 node에 해당하는 text들(**상품** 설명)을 BERT로 embedding
    - E: 각 node는 해당하는 text들(category 이름)을 BERT로 embedding→ 해당 embedding으로 E를 생성한 뒤에, 학습을 통해 **단순한 단어 뜻이었던 label vector**에서 **계층구조 문맥이 포함된 label vector**로 변환
    - 둘의 내적을 통해 상품 vector x와 가장 유사한 구조적으로 강화된 label vector E를 찾게 됨
    
    A: 단순히 input feature을 linear layer를 통과하고 그 결과를 label embedding과의 내적을 통해 가장 유사도가 높은 label이 실제 답과 얼마만큼의 유사도를 가지는지 확인
    
    - 정확도 : Acc: 0.5772 | F1-macro: 0.5067
    - 단순 input feature을 label embedding과 같은 차원으로 투영시켜 유사도 만들어내는 함수
        
        ```python
        # Classifier that uses label embeddings to make predictions
        class InnerProductClassifier(nn.Module):
            def __init__(self, input_dim, label_embeddings, trainable_label_emb=True):
                super().__init__()
                # Project input features into the same dimension as label embeddings
                self.proj = nn.Linear(input_dim, label_embeddings.size(1))
        
                if trainable_label_emb:
                    # Label embeddings are trainable parameters
                    self.label_emb = nn.Parameter(label_embeddings.clone())
                else:
                    # Label embeddings are fixed (not updated during training)
                    self.register_buffer("label_emb", label_embeddings.clone())
        
            def forward(self, x):
                # Project input feature vectors
                x_proj = self.proj(x)
                # Compute logits as similarity with each label embedding
                logits = torch.matmul(x_proj, self.label_emb.T)
                return logits
        ```
        
    
    GCN label embdding
    
    - 코드
        
        ```python
        # ==========================================================
        # Your Task: Implement Label GCN and GCN-Enhanced Classifier
        # ==========================================================
        
        class LabelGCN(nn.Module):
            """
            Multi-layer Graph Convolutional Network (GCN) encoder for label embeddings.
        
            Each layer should perform the following steps:
                1. Aggregate neighbor embeddings: H <- A_hat @ H
                2. Linear transformation: H <- H @ W
                3. (Optional) Apply ReLU and Dropout (skip for the last layer)
        
            Args:
                emb_dim (int): Dimension of label embeddings.
                num_layers (int): Number of GCN layers.
                dropout (float): Dropout probability.
            """
            def __init__(self, emb_dim, num_layers=1, dropout=0.5):
                super().__init__()
                # TODO: Define learnable weight matrices (list of emb_dim x emb_dim parameters)
                # Hint: Use nn.ParameterList and Xavier uniform initialization
                self.weights=nn.ParameterList(
                    [nn.Parameter(torch.empty(emb_dim,emb_dim))
                      for i in range(num_layers)
                    ]
                )
                self.num_layers=num_layers
                self.dropout=dropout
                for i in range(num_layers):
                  nn.init.xavier_uniform_(self.weights[i])
                
        
            def forward(self, H, A_hat):
                """
                Args:
                    H (torch.Tensor): Initial label embeddings, shape (num_labels, emb_dim).
                    A_hat (torch.Tensor): Normalized adjacency matrix, shape (num_labels, num_labels).
        
                Returns:
                    torch.Tensor: Updated label embeddings, shape (num_labels, emb_dim).
                """
                # TODO: Implement multi-layer GCN
                # for each layer:
                #   1) propagate messages: H = A_hat @ H
                #   2) linear transform: H = H @ W
                #   3) if not last layer: apply ReLU + Dropout
               
                for i in range(self.num_layers-1):
                  H=torch.matmul(A_hat,H) # 각 layer마다 A와 연산을 해줘야함
                  H=H@self.weights[i]
                  H=torch.nn.functional.relu(H)
                  H=torch.nn.functional.dropout(H,self.dropout)
        
                H=torch.matmul(A_hat,H)
                H=H@self.weights[self.num_layers-1]
        
                return H
        
        class GCNEnhancedClassifier(nn.Module):
            """
            Classifier that combines:
              - Document representations projected into label space
              - Label embeddings refined by a GCN over the label graph
        
            Args:
                input_dim (int): Dimension of input document embeddings.
                label_init_emb (torch.Tensor): Initial label embeddings, shape (num_labels, emb_dim).
                A_hat (torch.Tensor): Normalized adjacency matrix of labels, shape (num_labels, num_labels).
                num_layers (int): Number of GCN layers.
                dropout (float): Dropout probability.
            """
            def __init__(self, input_dim, label_init_emb, A_hat, num_layers=1, dropout=0.5):
                super().__init__()
                # TODO:
                # 1. Define projection layer (input_dim -> emb_dim)
                # 2. Define GCN encoder for label embeddings
                # 3. Make label_init_emb trainable (nn.Parameter)
                # 4. Register adjacency matrix (use register_buffer)
                self.W_projection=nn.Parameter(torch.empty(input_dim,label_init_emb.shape[1]))
                nn.init.xavier_uniform_(self.W_projection)
                self.GCN_encoder=LabelGCN(label_init_emb.shape[1],num_layers,dropout)
                
                # 단순히 tensor가 아닌 parameter로 만들어야함
                self.label_embeddings=nn.Parameter(label_init_emb)
        
                # parameter(학습 대상)이 아니지만, 모델과 함께 GPU로 이동해야하므로 register_buffer을 사용
                self.register_buffer('A_hat',A_hat)
        
                self.num_layers=num_layers
                self.dropout=dropout
                
        
            def forward(self, x):
                """
                Args:
                    x (torch.Tensor): Input embeddings for documents, shape (batch_size, input_dim).
        
                Returns:
                    torch.Tensor: Logits for classification, shape (batch_size, num_labels).
                """
                # TODO:
                # 1. Refine label embeddings using GCN
                # 2. Project input x into label embedding space (+ dropout)
                # 3. Compute similarity (inner product) between x_proj and label_emb
                label_embedding=self.GCN_encoder(self.label_embeddings,self.A_hat)
                
                x=x@self.W_projection
                
                x=torch.nn.functional.dropout(x,self.dropout)
                
                inner_product=x@label_embedding.T
                return inner_product
        ```
        
    </aside>
    

## GAT (Graph Attention Networks)

[arxiv.org](https://arxiv.org/pdf/1710.10903)

👎 GCN에서는 모든 neigbors를 모두 똑같이 중요하다고 가정

💡 몇몇 neighbors이 더 정보를 많이 가지고 있을 수 있음 ⇒ **using attention**

- 결국 어떤 network이든지 input으로 여러 vector을 받아 output으로 단일한 vector을 만드는 미분가능한 함수이기만 하면 상관이 없음!
- `attention` : 학습을 통해 vector의 가중치를 다르게 둔다
    - 😎 `attention` 은 특정 부분에 집중을 한다는 하나의 철학이고 Query, Key, Value의 Self-attention은 그것을 구현한 구체적인 방식!

![image.png](Leveraging%20Relations/image%2026.png)

🤔 구체적으로 그럼 어떻게 `attention` 을 이용할까? 

🅰️ small neural network를 만들어 importance를 계산

- 먼저 가중치 $\alpha_{vu}^{k}$를 먼저 구해야함
    - 학습 가능한 parameter가 있는 network `a`를 이용하여 두 node(v,u)간의 점수를 계산
    - v의 모든 neigbor node에 대해 점수를 계산 후 softmax를 통해 정규화(확률→ 가중치로 만듦
- 구한 가중치를 이용해 이전 layer의 neigbor의 영향력을 조절 (weight에 따라 다르게 영향)

이렇게 하나의 layer을 만들고, 이것을 K번 반복하여 K-layer network를 만들 수가 있다

![image.png](Leveraging%20Relations/image%2027.png)

<aside>
📖

small network a?

attention 연산을 진행하는 network a의 경우는 **학습이 가능한 parameter가 존재하는 어떠한 network든 상관이 없다!**

- 아래 그림과 같이 one-layer attention network도 가능!
- 기존 실제 논문에서는 multi-head attention등도 사용을 가능

gradient descent를 통해 다른 parameter들과 함꼐 학습이 진행 ⇒ loss를 가장 많이 줄이는 edge(특정 노드간의 관계)에 더 많은 attention을 부여하도록 학습이 진행

![image.png](Leveraging%20Relations/image%2028.png)

</aside>

Node classification 결과

![image.png](Leveraging%20Relations/image%2029.png)

## Stacking GNN layers

우리는 해당 neighbors들의 정보를 모아 원하는 depth(layer)로 network를 만들어 feature vector을 만들 수 있다

**👎 GNN layer을 늘렸을 때의 문제점**

1. **Overfitting**: layer가 늘어남에 따라 parameter가 더 많아지면 overfitting에 취약해짐 **(feature을 만들 때 overfitting 발생가능)**
    - `oversmoothing` 이 발생한 뒤 classification 문제에서는 대체적으로 underfitting이 일어나게 됨 (**overfitting이 아님**, 물론 특정 noise를 학습하여 classify을 할 수 있지만 이것은 너무 학습을 많이해서 발생하는 것이 아님) **(feature을 가지고 downstream task를 수행할 때)**
2. **⭐️ Over-smoothing**: node embedding이 점진적으로 구별 불가능해지고, 비슷한 값들로 수렴 ⇒ **우리는 각 node embedding이 node를 구별하기를 원함!**
    - layer을 쌓는 것 = node들의 정보를 **avg/mix** 하는 것과 같음
    - 😎 non-linear function을 빼고 보았을 때 **A의 반복적인 적용** ⇒ ⭐️**eigen decomposition을 해보면 eigenvalue<1인 성분이 K가 커질 수록 빠르게 0 ⇒ `low frequency` 성분만 남음** (node embedding이 유사)

`Receptive field` : target node(node of interest)의 embedding을 결정하는 노드들의 집합 (CNN에서 filter가 input에 적용되는 것과 같음)

1. layer이 늘어날 수록, 각 node의 receptive field는 급격히 늘어남 ⇒ **거의 모든 node에 정보를 이용**
2. layer이 늘어날 수록, 두 node가 동시에 공유하는 receptive field가 급격히 늘어남 ⇒ **두 node가 비슷한 정보를 가짐 → embedding이 유사해짐**

![image.png](Leveraging%20Relations/image%2030.png)

`paradox` : 본래 deep learning은 layer을 여러개둠으로써 표현력을 극대화시키는 방법인데, GNN에서는 성립이 되지 않아 deep learning의 본질을 활용하지 못하는 문제가 발생 

<aside>
📖

다른 모델(CNN..)에서 deep learning paradox가 생기지 않는 이유

결국 GNN의 문제는 `oversmoothing` , 즉 **평균화 연산을 여러번 누적을 하는 모델의 구조적 문제**

`🙋‍♀️ CNN` 에서 비슷한 문제가 생기지 않을까?

🅰️ `chat gpt` overfitting은 쉬워지지지만 CNN에서는 다양한 filter + pooling + skip 등을 이용하여 단순히 평균화 연산만을 진행하지 않고, Transformer는 전역 attention으로 직접 연결하므로 평균화 연산을 완화 

🅰️ `Gemini` : kernel 연산이 결국 neighborbood을 모으는 것과 같아지므로, ⭐️**CNN layer가 깊어질 수록 Receptive Field의 영역이 점점 커짐** ⇒ 너무 깊어지면 동일한 입력을 보고 계산 ⇒ **해결책도 residual connection으로 같음!**

🙋‍♀️ ResNet이 나온 이유는 layer을 깊게 쌓으면 **gradient vanishing/ exploding**이 일어나 성능이 떨어짐 (underfitting이 일어남) ⇒ **이미지 모델 구조(CNN..)은 결국 oversmoothing 문제가 큰 편은 아닌데 (input 자체 차원이 크니까 receptive field가 겹쳐지려면 많은 layer을 쌓아야함) , gradient vanishing과 함께 문제점이라고 볼 수 있나?**

🙋‍♀️만약 image의 차원이 줄어들어서 node의 개수와 비슷해지게 된다면 유사한 문제점이 나타날 수 있나?

  `MLP` 의 각 layer은 fully-connected이기 때문에 이웃이라는 개념 X⇒ 처음부터 전역적인 계싼을 수행

</aside>

### 어떻게 효율적인 GNN을 만들까?

먼저 단순하게 `oversmoothing` 문제가 생기지 않기 위해 graph의 receptive field에 대한 분석이 필요 ⇒ 그래프의 `diameter` (가장 멀리 떨어진 두 노드 상의 최단거리)

- 만약 layer의 수가 diameter보다 크면 모든 node가 서로에게 전달할 수 있음
- 이것들을 이용해 적당한 수의 layer을 고려해야함

👎 하지만 이러한 shallow network는 적은 expressive power를 가짐

- 그렇다면 어떻게 이러한 GNN의 expressive power을 높일 수가 있을까?

1.  **각 GNN layer안의 expressive power을 증가** ⇒ small neural network a를 더 깊은 network로 대체
    - 각 layer의 expressive power을 높여서 receptive field를 늘리지 않으면서도 전체 expressive power을 증가시킴
    - 😎 `CNN` 에서 `NiN(Network in Network` 와 유사 ⇒ `GoogLeNet` (1X1 → 3X3 → 1X1 conv)
2. **정보를 다른 node로 확산시키지 않는 layer을 추가 ⇒ MLP layer을 앞뒤로 붙임**
    - 모든 node가 동일한 MLP weight를 공유하지만, 계산은 각 node의 feature에 대해서만 개별적으로 수행 (노드 A feature → MLP → 노드 A의 새로운 feature - B의 정보 X)
    - 😎 `CNN`에서 1X1 conv와 유사 (이웃 pixel과 정보를 섞지 않고 차원정보만을 변환)
    - `preprocessing layer` : **raw node feature을 의미있는 feature로 변환**
    - `post-processing layer` : node embedding을 이용해 특정 task등을 할 때 필요로 함
    - 실제로, **이러한 layer을 추가하는 것이 성능향상에 도움이 됨**

![image.png](Leveraging%20Relations/image%2031.png)

1. **Skip connection 이용**

💡 GNN의 앞쪽 layer가 뒤쪽 layer보다 node 구분을 더 잘하기 때문에 `shortcut` 을 통해 **앞쪽 layer의 영향을 더 늘려 oversmoothing을 해결하자!**

1. **각 GNN layer마다 shortcut**

![image.png](Leveraging%20Relations/image%2032.png)

1. **맨 마지막 layer에 한꺼번에 shortcut 모임**
    - aggregation function은 주로 `weighted sum` 을 이용
    - 🙋‍♀️ 차원이 다 같아야하니까 표현력에 문제가 생길 수 있지 않을까(결국 의미있는 저차원 vector을 만들어야하는데 한번에 차원을 줄이면 제대로 학습이 어렵지 않나?)
    - 🙋‍♀️ 각 GNN layer마다 차원이 줄어드는데 aggregate할 때만 Weight multiplication을 통해 차원을 맞춰주나?

![image.png](Leveraging%20Relations/image%2033.png)

Skip connection은 shallow & deep model의 mixture을 만들어 냄 ⇒ N개의 skip connection은 곧 2^N의 가능한 path가 생김 → **embedding ensemble 효과를 만들어 냄**

![image.png](Leveraging%20Relations/image%2034.png)

# GNN application

## Pinterest

`Pinterest` : 사람이 만들어난 아이디어의 모음으로, `pin` 과 `board` 로 구성됨

- pin: visual bookmark(컨텐츠)
- board: themed collection of pins (pin들을 주제별로 묶은 폴더)

🤔 어떻게 관련이 있는 유사한 pin을 찾을까?

1. pin을 encoder을 이용해 vector로 만들어 가장 가까운 neighbors를 찾는다! ⇒ 👎 `theme` 을 고려하지 않음! (**우리가 원하는건 유사한 theme안에서 유사한 pin을 찾는 것!**)
2. GNN 이용하여 encoding! (CNN, BERT의 경우 relation을 무시)

![image.png](Leveraging%20Relations/image%2035.png)

GNN이 아닌 다른 model을 사용하게 되면 `theme`을 제대로 고려하지 못하는 문제점이 생기지만 GNN은 **query pin에 대해서 해당 theme에 유사한 다른 pin을 찾을 수 있게 됨**

![image.png](Leveraging%20Relations/image%2036.png)

## Fairness of prediction

예시: sensitive attribute (NBA 선수의 국적)이 예측(선수 연봉 측정)에 미치는 영향을 없애야함!

- 미국 선수의 경우, 인기도 등 때문에 연봉이 더 높게 측정이 될 수 있지만 (실제로 상관관계 존재) 이것이 **실제 객관적인 능력을 보여주는 지표가 되면 안됨! (실제 연봉협상이나 player scouting을 할 때)**
- 해당 상관관계를 model이 학습한다면 이는 해당 bias를 더 증폭하게 됨 ⇒ **⭐️ 국적에 대한 정보를 최대한 줄이면서 예측의 정확도는 유지시켜야함**
- 물론 이미 상관관계가 높은 것을 억지로 영향도를 줄여야하므로 예측 성능이 줄어들 수 밖에 없음

`Statistical Parity(SP)` : sensitive attribute에 대해서 model 예측이 independent한지를 측정하는 척도! 

- $\Delta_{sp}=|P(\hat y|s=0)- P(\hat y|s=1)|$
- **sensitive attritbute와 무관하게 model이 예측 →** `SP` 가 작음 (같은 능력일 때 국적에 따라 연봉이 다르면 안됨)

![image.png](Leveraging%20Relations/image%2037.png)

실험 결과를 보면 GNN 모델들이 성능은 좋지만 **SP가 높아 unfairness**하다는 것을 알 수 있음!

### Adversarial learning

결국 우리가 원하는건 sensitive attribute에 대해서 model 예측이 invariant 해야함

💡 이를 할 수가 있는게 `Adversarial Learning!` (**일종의 정보를 지운다는 개념**으로 보면 되기 때문에!)

- $L_c(\theta_G,\theta_c)$ : label classification loss로 실제 모델 예측을 진행 (**main loss**)
- $L_S(\theta_G,\theta_c)$: sensitive attribute classification loss로 senstivie attribute 예측을 진행 **(auxilary loss)**
    - $\theta_G$ : 해당 loss를 최소화하려고 함(**GRL**) → attribute invariance하게 만듦 **(GRL을 통해 해당 정보를 잊어버리게)**
    - $\theta_S$ : 해당 loss를 최대화하려고 함 → sensitive attribute 예측을 잘하게 **(의미 있는 학습 진행) ⇒ adversarial Game을 진행**

![image.png](Leveraging%20Relations/image%2038.png)

### Adversarial Learning with SSL

😢 하지만 실제 해당 방법을 적용하려고 할 때는 auxiliary loss를 만드는데 필요한 sensitive 정보들을 얻는 것이 쉽지 않음 (privacy)

💡 label scarcity ⇒ `Semi Supervised Learning` (Pseudo-labeling &. Self-training) 

![image.png](Leveraging%20Relations/image%2039.png)

1. 추가적인 모델을 만들어 해당 input이 어떤 label을 가질 것인지 예측 ⇒ `pseudo- labeling`
2. 해당 label을 가지고 adversarial learning loss를 통해 모델 학습을 진행 

이때 어떠한 pseudo label을 사용할지 정할 수 있음

1. **Hard label**: 추가적인 모델을 만들어서 생성하는 label(estimation)이 binary!
    - 👍 간단하고 해석에 용이
    - 👎 threshold 근처에서 error에 매우 민감 & 잘못된 pseudo-label이 전파될 수 있음 → unstable
2. **Soft label**: 추가적인 모델을 만들어서 생성하는 **label(estimation)이 확률적!**
    - 👍 ⭐️ loss가 **estimator의 confidence에 따라 가중치**를 주게 됨 ⇒ binary인 경우에는 equally weighted가 된다면, confidence가 크면 클 수록 모델이 해당 label을 확실하게 여겨 학습을 진행가능하게 됨
    - 👍 확률적으로 값을 매기기 때문에 estimation error의 덜 민감함
    - 👎 해석에 용이하지 않음 (pseudo label이 확률적으로 나옴)

대부분의 경우 `soft label` 을 사용하는 것이 더 선호됨

- pseudo label은 main task가 아닌 auxiliary loss (sensitive attribute 지우기)에 사용이 됨
- 일종의 confidence weighted signal 역할을 수행 → 확실하지 않은 node에 적은 기여를 함

⇒ 해당 방식을 이용해 정확도는 거의 일치하면서 SP를 줄이는데 성공을 했다!

![image.png](Leveraging%20Relations/image%2040.png)