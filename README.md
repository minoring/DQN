# Playing Atari with DQN

## (Working in Progress...)

This project implement DQN paper in Atari 2600 environment. The goal is to train an agent as similar as possible to original paper [DQN nature](https://www.nature.com/articles/nature14236) (such as preprocessing, frame skip, other hyperparameters... though there are some modification for practical reasons).

## Requirements
```
gym                 0.21.0
ale-py              0.7.3
torch               1.10.0
torchvision         0.11.1
```

## Results
## Video
| Breakout | Enduro | Space Invaders |
|:---: | :---: | :---: |
|![](assets/Breakout.gif) | ![](assets/Enduro.gif) | ![](assets/SpaceInvaders.gif) |


## Game Score
DQN agent were tranined for 10 milion frames. Scores for each game are average over 10 episodes.

After training, run
`python test.py --env {env_name} --trained-mode-path {trained_model_path} --record-video` to compute average game score and record video (e.g.`python test.py --env ALE/Enduro-v5 --trained-model-path trained_model/Enduro.pt --record-video`). Add `--render-mode human` to test in interactive environment. Flags are defined in `get_test_args` function in [`parse_utils.py`](parse_utils.py).


Game | DQN (std)
:---:|:---:
Breakout | 97.3 (76.9)
Enduro | 275.1 (61.0)
Space Invaders | 313.0 (118.4)



## ROMs
- Download Atari 2600 [roms](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html), unzip, place files below in [`ROMS`](ROMS) directory.
	- Breakout - Breakaway IV (Paddle) (1978) (Atari, Brad Stewart - Sears) (CX2622 - 6-99813, 49-75107) ~.bin
	-  Space Invaders (1980) (Atari, Richard Maurer - Sears) (CX2632 - 49-75153) ~.bin
	- Boxing - La Boxe (1980) (Activision, Bob Whitehead) (AG-002, CAG-002, AG-002-04) ~.bin
	- Pong - Video Olympics - Pong Sports (Paddle) (1977) (Atari, Joe Decuir - Sears) (CX2621 - 99806, 6-99806, 49-75104) ~.bin
	- Enduro - Enduro (1983) (Activision, Larry Miller) (AX-026, AX-026-04) ~.bin
- Run `ale-import-roms ROMS` to check every ROM is supported.
- Run `python env_test.py --env {ALE_ENV}` to check if you can create an environment successfully. e.g. `python env_test.py --env ALE/SpaceInvaders-v5`
- This project use 3 environments, but you can add more if you want. Note there are various distribution of same game (Boxing for example, "Boxing (Unknown) (PAL).bin", "Boxing (Dactari - Milmar).bin", "Boxing (1983) (CCE) (C-861).bin" ...), make sure it is supported by ALE. (FYI, I just placed distributions one by one in ROMS folder and ran `ale-import-roms ROMS` command to check whether it is supported).
- To get more infomation about ALE, check [docs](https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master/docs), [blog](https://github.com/mgbellemare/Arcade-Learning-Environment)

## Train
Run `python train.py --env CartPole-v0` to test if the algorithm works in relative simple environment.

## References
### Paper
- [DQN Nature](https://deepmind.com/research/publications/2019/human-level-control-through-deep-reinforcement-learning)
