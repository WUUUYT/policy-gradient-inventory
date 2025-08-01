import yaml
from envs.two_echelon_env import TwoEchelonEnv
from agents.reinforce_agent import ReinforceAgent
from utils.logger import Logger

def main():
    config = yaml.safe_load(open("config/default.yaml"))
    env = TwoEchelonEnv(config["env"])
    agent = ReinforceAgent(**config["agent"])
    logger = Logger("outputs/logs")
    
    for batch in range(config["train"]["num_batches"]):
        # reset per episode, collect log_probs+rewards, call agent.update(...)
        # logger.log(...)
    agent.save("outputs/checkpoints/final.pt")

if __name__ == "__main__":
    main()