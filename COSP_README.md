# Integrating SAVN into COS-POMDP codebase

Using the virtualenv of cos-pomdp, set up this code of savn.


1. Activate the cosp virtualenv

2. Instead of installing everything in `requirements.txt`, which
   contain a lot of outdated packages, just install

   ```
   pip install setproctitle
   pip install tensorboardX
   pip install h5py
   pip install tqdm
   ```

   And install torch, depending on your CUDA version. With CUDA 11.3:
   ```
   pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   This uses the latest (as of now) Pytorch version 1.9.0

3. Test out the repo by running this command:

    ```
    python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/savn_pretrained.dat \
    --model SAVN \
    --results_json savn_test.json

    cat savn_test.json
    ```
    Expected output:
    ```
    13%|███████▏                                              | 132/1000 [01:08<03:48,  3.79it/s]
    ```
    Interestingly, it just works on 3.3.4. Nothing needed to be done.
    Also, it is surprising to me that when this program runs,
    the Unity window of ai2thor stays static. How does he do that?

    I ended up getting this exception at 750/1000
    ```
    75%|████████████████████████████████████████▌             | 750/1000 [08:16<03:12,  1.30it/s]Traceback (most recent call last):
    ...
    posixsubprocess.fork_exec(
       OSError: [Errno 12] Cannot allocate memory
    ```
    It is fine for now. My goal here is to extract out an agent I can use for my experiments.
    Their code works - that means I should be able to.
