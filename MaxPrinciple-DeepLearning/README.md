# (Adv. Training Modification) Maximum Principle Based Algorithms for Deep Learning

*From Narsil*: 

After reading through Prof. Li's codes, I have made some small changes in `msalib.network.Network()` and 
`msalib.train.MSATrainer()` and then get `advNetwork` and `advMSATrainer` respectively.
  
In `main_toy.py` or `train_mnist.py`, we only need to use `advNetwork` and `advMSATrainer` to take place of their
original version, simultaneously specifying the shape of input when defining `advNetwork`, then somehow we can get an optimal
control version of adversarial training. I'm not sure about its feasibility but there seems no bug.

----
Tensorflow implementations of the E-MSA algorithm introduced in

*Maximum Principle Based Algorithms for Deep Learning.
Qianxiao Li, Long Chen, Cheng Tai, Weinan E.
Journal of Machine Learning Research 18 165:1–165:29. 2018*

URL: <http://jmlr.org/papers/v18/17-653.html>

## Requirements

`tensorflow`, `numpy`, `matplotlib`, `pyyaml`

## Examples

Run algorithm on test problems (configuration file in [config.yml](config.yml))

```python
    python -u main_toy.py | tee test.log
```

```python
    python -u main_mnist.py | tee mnist.log
```

Plot training curves

```python
    python plot_logs.py --logdir test.log
```

## License

This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details
