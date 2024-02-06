# AlphaZero
Simplest AlphaZero Implementation. Only two python files within a thousand lines:
- `game.py`: define the game logic (rules, valid move, game ended, score, etc.). We take Othello as an example, you can easily implement other games like gomoku, chess, Go.
- `alphazero.py`: MCTS, Self-Play, RL and Pytorch based Neural Networks.

This repo is based on https://github.com/suragnair/alpha-zero-general, while much more simpler:
- It only contains two python files, the only requirement is Pytorch (any version).
- Make the recursive MCTS more intuitive: the `search(canonicalBoard)` function return the value of the current canonicalBoard which is recursively called: `v = -search(next_canonicalBoard)` if current canonicalBoard isn't a terminal/leaf node based on the zero-sum game nature.
- To save people with OCD: `getGameEnded(self, board, player)` return None if not ended, 1 if player won, -1 if player lost, 0 if draw. Rather then a small positve number for draw.
- Move all the args into one place and make it configurable.
- Fix several bugs and erroneous comments.

## Train

```
python alphazero.py --train
```
One day training with one common GPU, you will go though ~100 iterations and get a strong Othello player. The default board size is 6 * 6, you can change it by `--board_size=8` to have a common 8 * 8 board.

I put my trained 6 * 6 ckpt under `temp/iter106.pth.tar`.
## Play
Random player vs. Alphazero: 0 : 100
```
python alphazero.py --play --round=100 --player1=random --ckpt_file=iter106.pth.tar
```

Greedy player vs. Alphazero: 0 : 100
```
python alphazero.py --play --round=100 --player1=greedy --ckpt_file=iter106.pth.tar
```

Alphazero vs. Alphazero: 0 : 1 (99 draws)
```
python alphazero.py --play --round=100 --player1=alphazero --ckpt_file=iter106.pth.tar
```
You can play with Alphazero too, please let me know if you can beat it:
```
python alphazero.py --play --round=2 --player1=human --ckpt_file=iter106.pth.tar --verbose
```

## Reference
- [Othello is Solved](http://mp.weixin.qq.com/s?__biz=Mzg2MTUyODU2NA==&mid=2247598677&idx=1&sn=5d9751b703b23b517320cf1d704753dd&chksm=ce16cacef96143d81815c5ddb2f8cad3de6de92d89d98db9c61ebda3cfa065236695d35834f6&mpshare=1&scene=24&srcid=1227fjlEsKIa8nOPps22AMlM&sharer_shareinfo=959b50970f470f989ff044475ff3d97d&sharer_shareinfo_first=959b50970f470f989ff044475ff3d97d#rd)
