import os, json
import lib.model as models

class setup():

    def __init__(self, env_id):

        with open(os.path.join('conf', env_id + '.json'), 'r+') as json_file:

            data = json.load(json_file)

            self.ENV_ID          = data.setdefault('env_id', 'RoboschoolHalfCheetah-v1')
            self.NUM_INPUTS      = data.setdefault('num_inputs', 26)
            self.NUM_OUTPUTS     = data.setdefault('num_outputs', 6)
            self.NUM_ENVS        = data.setdefault('num_envs', 1)
            self.HIDDEN_SIZE     = data.get('hidden_size', 256)
            self.LEARNING_RATE   = data.setdefault('learning_rate', 1e-4)
            self.GAMMA           = data.setdefault('gamma', 0.99)
            self.GAE_LAMBDA      = data.setdefault('gae_lambda', 0.95)
            self.PPO_EPSILON     = data.setdefault('ppo_epsilon', 0.2)
            self.CRITIC_DISCOUNT = data.setdefault('critic_discount', 0.5)
            self.ENTROPY_BETA    = data.setdefault('entropy_beta', 0.001)
            self.PPO_STEPS       = data.setdefault('ppo_steps', 256)
            self.MINI_BATCH_SIZE = data.setdefault('mini_batch_size', 64)
            self.PPO_EPOCHS      = data.setdefault('ppo_epochs', 10)
            self.TEST_EPOCHS     = data.setdefault('test_epochs', 10)
            self.NUM_TESTS       = data.setdefault('num_tests', 10)
            self.TARGET_REWARD   = data.setdefault('target_reward', 2500)
            self.MODEL_NAME      = data.setdefault('model_name', 'ActorCritic')
            self.MODEL_CLASS     = getattr(models, self.MODEL_NAME)

            # Transformations
            if isinstance(self.NUM_INPUTS, list): self.NUM_INPUTS = tuple(self.NUM_INPUTS)

            json_file.seek(0)  # go to beggining of file
            json.dump(data, json_file)  # write content
            json_file.truncate()  # clear any tail of old content

