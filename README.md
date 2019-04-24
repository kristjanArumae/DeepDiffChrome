# DeepDiffChrome

[DeepDiff: Deep-learning for predicting Differential
gene expression from histone modifications](https://academic.oup.com/bioinformatics/article/34/17/i891/5093224)

## Training Model
To train, validate and test the model for celltypes "Cell1" and "Cell2": 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python train.py --cell_1=Cell1 --cell_2=Cell2  --model_name=raw_d --epochs=120 --lr=0.0001 --data_root=data/ --save_root=Results/


### Other Options
1. To specify DeepDiff variation: \
--model_name= \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_d: difference of HMs \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_c: concatenation of HMs \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw: raw features- difference and concatenation of HMs \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_aux: raw features and auxiliary Cell type specific prediction features \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;aux: auxiliary Cell type specific prediction features \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;aux_siamese: auxiliary Cell type specific prediction features with siamese auxiliary \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;raw_aux_siamese: raw features and auxiliary Cell type specific prediction features with siamese auxiliary 

2. To save attention maps: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;use option --save_attention_maps : saves Level II attention values in .txt file 

3. To change rnn size: \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--bin_rnn_size=32 

4. Parameters specific to transformer extension: \
--transformer : Whether to use Transformer Encoder (implemented for raw_d, raw_c, and raw).\
--num_heads : Number of transformer heads.  Must divide evenly into --bin_rnn_size\
--num_t : number of transformers to stack.\
--norm : Whether to normalize input for transformer.  Cannot be used with LSTM.
