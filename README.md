<h2 align="center">
  SVAMP
</h2>
<h5 align="center"> Are NLP Models really able to Solve Simple Math Word Problems?</h5>

<p align="center">
  <a href="https://2021.naacl.org/"><img src="https://img.shields.io/badge/NAACL-2021-blue"></a>
  <a href="https://arxiv.org/abs/2006.09286"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://github.com/arkilpatel/SVAMP/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>



Transformers are being used extensively across several sequence modeling tasks. Significant research effort has been devoted to experimentally probe the inner workings of Transformers. However, our conceptual and theoretical understanding of their power and inherent limitations is still nascent. In our paper, we analyze the computational power as captured by Turing-completeness. 

<h2 align="center">
  <img align="center"  src="./images/trans-maskx.png" alt="..." width="350">
</h2>

We first provide an alternate and simpler proof to show that vanilla Transformers are Turing-complete given arbitrary precision. More importantly, we prove that Transformers with positional masking and without positional encoding are also Turing-complete. Previous empirical works on Machine Transation have found that Transformers with only positional masking achieve comparable performance compared to the ones with positional encodings. Our proof for directional transformers implies that there is no loss of order information if positional information is only provided in the form of masking. The computational equivalence of encoding and masking given by our results implies that any differences in their performance must come from differences in learning dynamics. 



<h2 align="center">
  <img align="center"  src="./images/transx.png" alt="..." width="300">
</h2>
We analyze the necessity of various components such as self-attention blocks, residual connections and feedforward networks for Turing-completeness. Interestingly, we find that a particular type of residual connection is necessary. Figure 2 provides an overview. We explore implications of our results on machine translation and synthetic tasks.



#### Dependencies

- compatible with python 3.6
- dependencies can be installed using `requirements.txt`

#### Setup

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Install all the required packages:

at `SAN:`

```shell
$ pip install -r requirements.txt
```

To create the relevant directories, run the following command in the corresponding directory of that model:

for eg, at `SAN/Transformer:`

```shell
$ sh setup.sh
```

#### Models

The current repository includes 2 implementations of Transformer Models:

- `Transformer`
  - Vanilla Transformer with options to remove residual connections
- `DiSAN`
  - Directional Self-Attention Network

#### Datasets

We work with the following datasets:

- `Copy_n` (Used for Copy Task)

  - Sentences  of lengths between n and n+5 are sampled from Penn Tree Bank Corpus for both train and test data.
  - `Train Data Size:` 43,199 sentences
  - `Validation Data Size:` 1000 sentences

- `count_data`

  - Numbers between 1-100 are mapped to the corresponding next five consecutive numbers.
  - Source input consists of a single number between 1-100 in the form a string.
  - Target output consists of a space separated string with a sequence of 5 numbers. 
  - `Train Data Size:` 1000
  - `Validation Data Size:` 100

  

#### Usage:

The set of command line arguments available can be seen in the respective `args.py` file.

##### Running Vanilla Transformer on Copy Task

at `./Transformer`:`

```
$	python -m src.main -gpu 0 -max_length 60 -batch_size 128 -epochs 50 -dataset copy_12 -run_name run_copy_task
```

##### Running Transformer without Encoder-Decoder Residual Connection on Copy Task

at `./Transformer:`

```
$	python -m src.main -gpu 0 -max_length 60 -batch_size 128 -epochs 50 -no-enc_dec_res -dataset copy_12 -run_name run_copy_task_wo_e-d-res
```

##### Running Transformer without Decoder-Decoder Residual Connection on Copy Task

at `./Transformer:`

```
$	python -m src.main -gpu 0 -max_length 60 -batch_size 128 -epochs 50 -no-dec_dec_res -dataset copy_12 -run_name run_copy_task_wo_d-d-res
```

##### Running Vanilla Transformer on Counting Task

at `./Transformer:`

```
$	python -m src.main -gpu 0 -max_length 6 -batch_size 32 -epochs 20 -dataset count_data -run_name run_counting_task
```

##### Running Transformer without Encoder-Decoder Residual Connection on Counting Task

at `./Transformer:`

```
$	python -m src.main -gpu 0 -max_length 6 -batch_size 32 -epochs 20 -no-enc_dec_res -dataset count_data -run_name run_counting_task_wo_e-d-res
```

##### Running Transformer without Decoder-Decoder Residual Connection on Counting Task

at `SAN/Transformer:`

```
$	python -m src.main -gpu 0 -max_length 6 -batch_size 32 -epochs 20 -no-dec_dec_res -dataset count_data -run_name run_counting_task_wo_d-d-res
```

##### Running SAN on Extended Copy Task

For example the data used is copy_16_20.

at `./Transformer:`

```
$	python -m src.main -gpu 0 -dataset copy_16_20 -run_name run_san_copy_16_20
```

##### Running DiSAN on Extended Copy Task

For example the data used is copy_16_20.

at `./DiSAN:`

```
$	python -m src.main -gpu 0 -dataset copy_16_20 -run_name run_disan_copy_16_20
```



For any clarification, comments, or suggestions please contact [Satwik](https://satwikb.com/) or [Arkil](http://arkilpatel.github.io/).