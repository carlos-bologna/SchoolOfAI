import argparse
import gym
from lib.environment import atari_env
#import roboschool

from lib.model import ActorCritic

import numpy as np
import torch
from lib.config import setup

ENV_ID = "RoboschoolHalfCheetah-v1"
HIDDEN_SIZE = 256

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=ENV_ID, help="Environment name to use, default=" + ENV_ID)
    parser.add_argument("-d", "--deterministic", default=True, help="enable deterministic actions")
    parser.add_argument("-r", "--record", help="If specified, sets the recording dir, default=Disabled")
    args = parser.parse_args()

    deterministic = args.deterministic
    
    # Setup all constant
    conf = setup(args.env)

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    #env = gym.make(args.env)
    env = atari_env(conf.ENV_ID)

    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    num_inputs = conf.NUM_INPUTS #envs.observation_space
    num_outputs = conf.NUM_OUTPUTS #envs.action_space

    #model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    model = conf.MODEL_CLASS(num_inputs, num_outputs, hidden_size=conf.HIDDEN_SIZE).to(device)

    model.load_state_dict(torch.load(args.model))

    state = env.reset()
    done = False
    total_steps = 0
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        if isinstance(dist, torch.distributions.categorical.Categorical):
            action = np.argmax(dist.probs.detach().cpu().numpy()) if deterministic \
                else int(dist.sample().cpu().numpy())

        elif isinstance(dist, torch.distributions.normal.Normal):
            action = dist.mean.detach().cpu().numpy()[0] if deterministic \
                else dist.sample().cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        total_steps += 1

    env.env.close()
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
