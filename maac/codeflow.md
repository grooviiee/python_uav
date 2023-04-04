reference..
https://github.com/shariqiqbal2810/MAAC
 -> https://github.com/shariqiqbal2810/multiagent-particle-envs 에서 environement를 구현했다.

requirements..
Python 3.6.1 (Minimum)
OpenAI baselines, commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
My fork of Multi-agent Particle Environments
PyTorch, version: 0.3.0.post4
OpenAI Gym, version: 0.9.4
Tensorboard, version: 0.4.0rc3 and Tensorboard-Pytorch, version: 1.0 (for logging)


<MAAC>
* Critic Network에 Attention Head가 존재한다. (Multi-attention head)
* attend_heads가 attention head의 수

path: main.py에서 진행 -> 실제 알고리즘은 attention_sac.py에서 실행

- 알고리즘 동작 순서
    sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
	model.update_critic(sample, logger=logger)
	model.update_policies(sample, logger=logger)
	model.update_all_targets()
