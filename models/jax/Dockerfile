FROM mariogeiger/e3nn-jax:0.6.3-jax0.3.14-cuda11.6.2

COPY functions.py /home/functions.py
COPY model.py /home/model.py
COPY evaluate.py /home/evaluate.py
COPY diffeomorphism.py /home/diffeomorphism.py
#COPY main.py /home/main.py

COPY args.pkl /home/args.pkl
COPY w.pkl /home/w.pkl

CMD ["python3", "/home/evaluate.py", "--path_args", "/home/args.pkl", "--path_weights", "/home/w.pkl", "--path_scripts", "/home", "--path_images", "/images", "--path_output", "/output"]

# To Build:
#   place this Dockerfile in the directory of the trained model
#   $ docker build -t instance2022-jax .

# To Run:
#   $ docker run --gpus all -e CUDA_VISIBLE_DEVICES=7 -v $(pwd)/images:/images -v $(pwd)/output:/output instance2022-jax
