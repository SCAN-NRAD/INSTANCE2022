
To train the network, we use the script `main.py`. The training loop is defined in `functions.py`. The configuration files are in the folder `configs/`. We used `submit2.py` for the last submission. The files for the submission are in the directory `submit_docker/`.

Command used to train the networks for the submission:

```
python main.py --name mynetwork0 --config configs/submit2.py --config.seed_init=0 --config.seed_train=0
python main.py --name mynetwork1 --config configs/submit2.py --config.seed_init=1 --config.seed_train=1
python main.py --name mynetwork2 --config configs/submit2.py --config.seed_init=2 --config.seed_train=2
python main.py --name mynetwork3 --config configs/submit2.py --config.seed_init=3 --config.seed_train=3
python main.py --name mynetwork4 --config configs/submit2.py --config.seed_init=4 --config.seed_train=4
python main.py --name mynetwork5 --config configs/submit2.py --config.seed_init=5 --config.seed_train=5
python main.py --name mynetwork6 --config configs/submit2.py --config.seed_init=6 --config.seed_train=6
```

Command used to evaluate? see the file `submit_docker/Dockerfile`.
