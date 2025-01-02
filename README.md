## NEWS
### AIGC platform
We developed an AIGC platform based on this project!

Welcome to the AI Music Generation test website:

http://www.rhythmelec.com/

## Overview
Official Codebase for POP-DIFFUSEQ: CONTROLLABLE SYMBOLIC MUSIC MULTI-INSTRUMENT INFILLING AND ACCOMPANIMENT GENERATION WITH LONG-AXIS ATTENTION

The main function is controllable and interactive symbolic(MIDI) music infilling and generation.
The input and output of Pop-Diffuseq are:

<p align = "center">
<img src="img/text-to-scores.jpg" width= "60%" alt="" align=center />
</p>
<p align = "center">
Accompaniment Generation under Text Prompts.
</p>

The diffusion process of our conditional framework.
<p align = "center">
<img src="img/framework.jpg" width= "80%" alt="" align=center />
</p>
<p align = "center">
The framework and diffusion process of Pop-Diffuseq.
</p>

## Data and trained model
The trained models, datasets and all unfiltered samples can be downloaded in Google Drive:
https://drive.google.com/drive/folders/1OWI_sfYmYn2gB13KccO9kkXCP2x8Z-1x?usp=drive_link

Due to the high economic cost of our self-built data set PopBand3k, 
we only disclosed part of the data demonstration during the review stage. 
Once the paper is accepted, we will release all the data for free immediately.

## Characteristics
The implement baseline are DiffuSeq:
[*__*DiffuSeq*__: Sequence to Sequence Text Generation With Diffusion Models*](https://arxiv.org/abs/2210.08933).

We improve the basic DiffuSeq in several methods:
(1) long-axis attention;
(2) structure-aligned representation;
(3) dynamic masking;
(4) multi-head output module.

The computational process of long-axis attention algorithm are:

The diffusion process of our conditional framework.
<p align = "center">
<img src="img/la-attention.jpg" width= "80%" alt="" align=center />
</p>
<p align = "center">
computational process of long-axis attention
</p>

The technical supplement is as follows:

https://github.com/musicai-cakecake/pop-diffuseq

## Objective evaluation:
Model setting:

(1) SA: Our proposed Structure-Aligned representation.

(2) MMR: Multi-track Multi-instrument Repeatable representation proposed by SymphonyNet, it is the baseline method of our representation.

(3) DiffuSeq: Discrete diffusion model(Version 2) for text generation, the baseline framework of our model.

(4) la-attention: Our proposed long-axis attention.

(5) linear transformer: Commonly used efficient attention for music generation tasks.

perplexity results:

<p align = "center">
<img src="img/ppl01.jpg" width= "60%" alt="" align=center />
</p>
<p align = "center">
PPL curve for total stage
</p>

<p align = "center">
<img src="img/ppl02.jpg" width= "60%" alt="" align=center />
</p>
<p align = "center">
PPL curves of all settings in the stable phase
</p>

## Setup:
The code is based on PyTorch and HuggingFace `transformers`.
```bash 
pip install -r requirements.txt
```
## Preprocessing

midi preprocessing: music-representation/lmd_accom_preprocessing.py

midi to token(representation): music-representation/accompaniment_generate_representation_lmd.py

token to midi: music-representation/sequence_to_midi_lmd.py

chord extract: music-representation/chords_detector

## DiffuSeq Training
```bash
cd scripts
bash train.sh
```
Arguments explanation:
- ```--dataset```: the name of datasets, just for notation
- ```--data_dir```: the path to the saved datasets folder, containing ```train.jsonl,test.jsonl,valid.jsonl```
- ```--seq_len```: the max length of sequence $z$ ($x\oplus y$) Default 512
- ```--resume_checkpoint```: if not none, restore this checkpoint and continue training
- ```--vocab```: the tokenizer is initialized using bert or load your own preprocessed vocab dictionary

### Update: Additional argument

- ```--learned_mean_embed```: set whether to use the learned soft absorbing state.
- ```--denoise```: set whether to add discrete noise
- ```--use_fp16```: set whether to use mixed precision training
- ```--denoise_rate```: set the denoise rate, with 0.5 as the default

```bash
cd scripts
bash run_decode_solver.sh
```

