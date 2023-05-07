# reference: https://github.com/rlsotlr01/PPO_practice/blob/master/main.py

import gym
import torch
from torch.distribution import Categorical
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# PPO Class
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []	# array storing status after a single step
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 3	# num_epoch

        # set neural network
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)

    # PPO - Policy, Value Network 코드
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):	# 그동안 쌓아온 값들을 batch에다 보내고 난 뒤에, flush
        s_list, a_list, r_list, s_prime_list, prob_a_list, done_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for item in self.data:
            # data로 쌓은 정보들을 각 배열에 저장
            s, a, r, s_prime, prob_a, done = item
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            prob_a_list.append([prob_a])
            done_mask = 0 if done else 1
            done_list.append([done_mask])

        # tensor로 값을 저장 후, 기존 data 배열은 flush
        s, a, r, s_prime, done_mask, prob_a = (
            torch.tensor(s_list, dtype=torch.float),
            torch.tensor(a_list),
            torch.tensor(r_list),
            torch.tensor(s_prime_list, dtype=torch.float),
            torch.tensor(done_list, dtype=torch.float),
            torch.tensor(prob_a_list),
        )
        self.data = []
        
        return s, a, r, s_prime, done_mask, prob_a

    def train(self):	# Start training
        # step 동안 쌓아둔 data들을 토대로 batch data를 만든다.
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # batch data를 가지고 epoch (=: K) 수만큼 반복해서 학습
        for i in range(self.K):
            td_target = r + self.gamma * self.v(s_prime) * done_mask

            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_list = []
            advantage = 0.0

			# GAE 값 산출
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])

            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)

            pi_a = pi.gather(1, a)	# gather를 통해서 	1차원의 idx만 추출

            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            # surrogated clipping
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                td_target.detach(), self.v(s)
            )

            self.optimizer.zero_grad()

            loss.mean().backward()

            self.optimizer.step()
            # end of epoch loop


def main():
    env = gym.make("CartPole-v1")  # Gyme에서 CartPole을 학습시킬거다.

    model = PPO()  # Model을 PPO로 설정
    gamma = 0.09  # Learning rate 설정
    num_step = 20

    score = 0.0
    print_interval = 20  # 20회 episode마다 출력
    scores = []
    num_episode = 1000

    # 10000번의 에피소드를 돌린다.
    for episode in range(num_episode):
        state = env.reset()
        done = False

        while not done:
            # STEP 1. T Step 만큼 게임을 진행 후에 training 진행
            for step in range(num_step):
                # STEP 1.1. def PPO.pi policy(critic): 상태 s를 넣으면 행동확률분포를 돌려준다.
                prob = model.pi(torch.from_numpy(state).float())
                # STEP 1.2. Categorical : 확률 값을 Categorical 데이터로 변환해준다.
                # e.g. [0.6, 0.3, 0.1] -> Categorical([0.6, 0.3, 0.1])
                m = Categorical(prob)
                
                # STEP 1.3. Categorical(tensor값).sample().item()을 하면 행동 분포 중에서 가장 확률이 높은 index를 돌려준다.
                # Categorical([0.6, 0.3, 0.1]) -> index 0 (확률 가장 높은 행동 출력)
                action = m.sample().item()

                # STEP 1.4. action a일 때 next_state와 reward, 게임종료 여부 등을 받아낸다.
                s_prime, reward, done, info = env.step(action)

                # STEP 1.5. replay buffer에다가 결과 데이터를 집어 넣는다.
                model.put_data(
                    (state, action, reward / 100.0, s_prime, prob[action].item(), done)
                )

				# STEP 1.6. 다음 step으로 동작할 준비.
                # 다음 상태를 기존 상태변수에 넣는다.
                state = s_prime
                # 에이전트의 점수칸에 보상을 더해준다.
                score += reward

                if done:
                    break

            # STEP 2. step T까지 진행 후, train 실행
            model.train()

        if episode % print_interval == 0 and episode != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    episode, score / print_interval
                )
            )
            scores.append(score)
            score = 0.0
    plt.plot(scores)
    plt.show()
    plt.savefig("result_main1.png", dpi=300)
    env.close()


if __name__ == "__main__":
    main()
