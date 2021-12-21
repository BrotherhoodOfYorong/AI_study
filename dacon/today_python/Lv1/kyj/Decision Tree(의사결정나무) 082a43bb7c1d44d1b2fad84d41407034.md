# Decision Tree(의사결정나무)

# 의사결정나무

어떠한 기준을 통해 데이터를 분류 또는 회귀하는 지도학습 모델 중 하나

결과 모델이 Tree 구조를 가짐

![Untitled](Decision%20Tree(%E1%84%8B%E1%85%B4%E1%84%89%E1%85%A1%E1%84%80%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE)%20082a43bb7c1d44d1b2fad84d41407034/Untitled.png)

# Decison Tree 원리

여러 개의 feature들 중 특정한 하나의 feature를 정해 이를 기준으로 모든 행을 두개의 노드로 분류할 수 있음

특정한 값을 정하는 decision tree의 대원칙은 "한쪽 방향으로 쏠리도록" 임. 하나의 decision tree를 만들어 봤지만, feature를 어떻게 배치하는 지에 따라 여러 decision tree를 만들 수 있고, 성능 또한 달라짐. 따라서 어떤 노드에 어떤 feature를 넣을지 선택해야하며 이를 위해 **불순도와 **향상도를 계산해서 찾아냄

# Decision Tree 구조

- 뿌리 노드 (root node)
- 중간 노드 (intermediate node): input 데이터를 분류할 수 있는 특성(feature)
- 브랜치(branch): 해당 feature이 가질 수 있는 value
- 최종 노드 (terminal node): 분류 결과(output)

![Untitled](Decision%20Tree(%E1%84%8B%E1%85%B4%E1%84%89%E1%85%A1%E1%84%80%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE)%20082a43bb7c1d44d1b2fad84d41407034/Untitled%201.png)

### Cost function: inforamtion gain (+ Entropy)

Entropy ⇒ 정보이론에서 우리가 가지고 있지 않은 정보의 양. 즉, 우리가 얼마나 이 시스템에 대해 혹은 변수에 대해 모르고 있는지, 얼마나 실제 관측이 우리를 놀라게 할 수 있는지에 대한 값

낮은 엔트로피 = 순도가 높은 상태 = 불순도가 낮은 상태

따라서, decision tree에서도 불순도를 줄이는 방식(아래로 내려갈수록 데이터의 엔트로피가 낮아지는 방식)으로 feature들을 어느 노드에 배치할 지를 정함

### Model 학습법: greedy algorithm

decision tree는 information gain이라는 measure를 통해, 현재 상황에서 inforamtion gain이 가장 높은 feature를 현재 노드로 선택해나가는 greedy algoritm을 통해 학습됨.

# Hyperparameters tuning

### stopping Criteria

학습을 하다가 특정 시점에서 멈추는 규칙

overfitting 과 해석이 어려워지는 것을 방지하기 위해 사용

1. 최대 깊이 (max_depth)
최대로 내려 갈 수 있는 depth.
작을 수록 트리가 작아짐
2. 최소 노드크기 (min_samples_split)
노드를 분할하기 위한 데이터 수.
해당 노드에 이 값보다 작은 확률 변수가 있다면 stop.
작을수록 트리 작아짐
3. 최소 향상도 (min_impurity_decrease)
노드를 분할하기 위한 최소 향상도.
향상도가 설정값 이하라면 더 이상 분할하지 않음.
작을수록 트리 커짐
4. 비용 복잡도(cost-complexity) 함수

![Untitled](Decision%20Tree(%E1%84%8B%E1%85%B4%E1%84%89%E1%85%A1%E1%84%80%E1%85%A7%E1%86%AF%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%82%E1%85%A1%E1%84%86%E1%85%AE)%20082a43bb7c1d44d1b2fad84d41407034/Untitled%202.png)

- |T|: 최종노드의 개수
- 𝛼: 패널티 계수 (트리가 커지는 것에 대한 패널티를 사용자가 정함)
    각 𝛼 값마다, 가장 Cost 를 적게 하는 트리 T 가 존재함
- 𝛼=0 이라면, 최대크기의 트리가 가장 선호됨 (0.5 면 뿌리노드)

    𝛼 값이 작으면 train 데이터에 대해서는 정확도가 높지만, test 데이터에 대해서는 정확도 낮음

    최적의 𝛼 값은 교차타당성(Cross-Validation) 을 통해 찾을 수 있음

출처

[https://github.com/pyohamen/Im-Being-Data-Scientist/wiki/what-is-decision-tree%3F](https://github.com/pyohamen/Im-Being-Data-Scientist/wiki/what-is-decision-tree%3F)

[https://sanghyu.tistory.com/8](https://sanghyu.tistory.com/8)

[https://github.com/pyohamen/Im-Being-Data-Scientist/wiki/Hyperparameters-tuning](https://github.com/pyohamen/Im-Being-Data-Scientist/wiki/Hyperparameters-tuning)