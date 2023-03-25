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
        self.data = []
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps = 0.1
        self.K = 3

        # 신경망 구축
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

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, prob_a_list, done_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for item in self.data:
            s, a, r, s_prime, prob_a, done = item
            s_list.append(s)

            a_list.append([a])
            r_list.append([r])

            s_prime_list.append(s_prime)
            prob_a_list.append([prob_a])
            done_mask = 0 if done else 1
            done_list.append([done_mask])

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

    def train(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # 같은 batch data에 대해서 epoch 수만큼 반복해서 학습
        for i in range(self.K):
            td_target = r + self.gamma * self.v(s_prime) * done_mask

            delta = td_target - self.v(s)
            # TODO: detach의 의미와 하는 이유를 찾아볼 것
            delta = delta.detach().numpy()

            advantage_list = []
            advantage = 0.0

            for delta_t in delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_list.append([advantage])

            advantage_list.reverse()
            advantage = torch.tensor(advantage_list, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)

            pi_a = pi.gather(1, a)

            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

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
    env = gym.make("CartPole-v1")

    model = PPO()

    gamma = 0.09

    T = 20

    score = 0.0
    print_interval = 20
    scores = []

    # Main 2 code
    # 10000번의 에피소드를 돌린다.
    for n_epi in range(1000):
        # 환경을 리셋하여 초기 상태를 얻는다.
        s = env.reset()
        # 게임 종료 여부를 False로 초기화
        done = False
        # 게임이 끝날 때까지 돌림
        while not done:
            # T만큼 게임을 진행.
            for t in range(T):
                # PPO.pi - 정책 : 상태 s를 넣으면 행동확률분포를 돌려준다.
                prob = model.pi(torch.from_numpy(s).float())
                # Categorical : 확률 값을 Categorical 데이터로 변환해준다.
                # e.g. [0.6, 0.3, 0.1] -> Categorical([0.6, 0.3, 0.1])
                m = Categorical(prob)
                # Categorical([0.6, 0.3, 0.1]) -> index 0 (확률 가장 높은 행동 출력)
                a = m.sample().item()
                # Action a일 때 State'와 Reward, 게임종료 여부 등을 받아냄.
                s_prime, r, done, info = env.step(a)

                # 데이터를 집어 넣는다.
                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))

                # 다음 상태를 기존 상태변수에 넣는다.
                s = s_prime
                # 에이전트의 점수칸에 보상을 더해준다.
                score += r

                if done:
                    break

            # T까지 진행 후, train 실행
            model.train()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{}, avg score : {:.1f}".format(
                    n_epi, score / print_interval
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
