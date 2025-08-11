# RL Playground

This repository consists of some DeepRL algorithms that I implemented myself and some meta-learning stuff that I am interested in. Feel free to have a look around! For the main part, I am roughly following OpenAI's [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/) and made this little playground to test my implementations quickly.


### Setup (Python 3.10.15, M3 MacBook Air)

1. `python -m venv env`
2. `source env/bin/activate`
3. `pip install -r requirements.txt`

You can set up experiments using config.yaml. Simply change the parameters to the ones you want to try! I structured the envs in this rather unconventional way to facilitate some random parametrization for meta-learning approaches that are still in development. 
