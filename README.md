"# rl"

My first python simulater implementation.

Implementation course:

1. Deep Learning -> v
2. Q-learning -> v
3. Deep Q-learning -> v
4. REINFORCE -> v
5. Actor-critic -> v
6. DDPG

Implementing MultiAgent-PPO
Next step) Attention based actor-critic
Next step) Graph based Attention Actor Critic

Next Next step) Consider transformer based Reinforcement learning method. (Can this method be the trend?)
Transformer를 적용해서 디자인한 RL 알고리즘: "Decision Transformer: Reinforcement Learning via Sequence Modeling"
https://hugrypiggykim.com/2022/03/29/decision-transformer-reinforcement-learning-via-sequence-modeling/ 에 해당 논문에 대한 설명이 있다.
이를 참고해서 공부해볼 것

Attention based PPO --> Please refer git hub repository "https://github.com/RvuvuzelaM/self-attention-ppo-pytorch"

to do:
1. 어탠션 알고리즘 구현 완료
구현 시에 actor critic network 세팅을 합치기
2. 리워드 계산 방식 검증
3. 네트워크 rnn 검증. (input과 output setting들)

on going) matplot lib 붙이기

Item.
AoI (Age of Information): Timeliness index to characterize the freshness of contents.
QoE (Quality of Experience): Measures CDI(Content Delay Index) which is formulated based on the latency involved in the delivery of contents requested by ground-based mobile users.
 -> CDI라는 matric을 사용하는데, Delay 산출 방식과 유사하다.

23.11.13
-> 모두 Delay 관련된 matric을 최적화하는 Item인데, 그러면 기존 reference 논문과 비슷하지 않나??
-> 후속논문으로 다른 item들이 있는지도 확인해봐야겠다.

-> 만약에 User가 움직이는 환경이면 뭔가 달라질 수 있을까??


24.02.27
-> 기존 알고리즘에 대한 검증 필요
-> IAB에 대한 리서치가 필요할 것 같다. (연구 주제와 겹칠 우려가 발생)

24.03.07
-> 알고리즘 구현이 너무 복잡한 것 같아서 갈아타야 겠다는 생각이 들었음
(내가 구현을 했는데, 디버깅이 어려울 것 같아서..)
-> action_space.n이 여러개인 환경에 대해서 학습이 좀 더 필요해보인다. 이게 맞나...?
