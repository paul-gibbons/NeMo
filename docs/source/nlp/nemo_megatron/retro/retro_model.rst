NeMo RETRO Model
================

The Retrieval-Enhanced Transformer (RETRO) model is an autoregressive language model that takes into account document chunks retrieved from a large 
corpus when making predictions. The RETRO model has a similar architecture to the GPT model, but it includes an encoder that encodes the retrieved 
context and cross-attention layers that integrate the context to improve the model's output. Below is a simple diagram of the RETRO model architecture.

.. image:: images/arch.png
    :align: center
    :width: 800px
    :alt: RETRO model architecture

For more detailed information on the model, please refer to the `RETRO paper <https://arxiv.org/abs/2112.04426>`_ :cite:`nlp-retro-borgeaud2021improving` by Deepmind. 
The NeMo RETRO Model is an open-source implementation of the paper, and it has the following differences/features compared to Deepmind's proposed implementation:

1. The NeMo RETRO Model is built on top of NeMo Megatron code, allowing for efficient training of large language models in a cluster environment.
2. The NeMo RETRO Model uses `Faiss <https://github.com/facebookresearch/faiss>`_ :cite:`nlp-retro-jegou2022faiss` as the K$N search library, which can be accelerated by GPUs. 
3. The NeMo RETRO uses `RoPe relative positional encoding <https://arxiv.org/abs/2104.09864>`_ :cite:`nlp-retro-su2021roformer`. 
4. The NeMo RETRO uses `SentenceTransformers <https://www.sbert.net>`_ :cite:`nlp-retro-reimers2019sentence` as the retriever encoder.
5. The NeMo RETRO supports `mu-Transfer <https://openreview.net/pdf?id=Bx6qKuBM2AD>`_ :cite:`nlp-retro-yang2022tensor`, allowing for scalable training of the RETRO model via Zero-Shot Hyperparameter Transfer.

Quick start
************
Steps below demonstrate training and evaluating a NeMo RETRO model

Data pre-processing
-------------------

For the data preparation step, we refer to `Megatron-LM Github <https://github.com/NVIDIA/Megatron-LM/>`_ repository, which contains the scripts and detailed instructions of the whole preprocessing process at `RETRO Data Preparation <https://github.com/NVIDIA/Megatron-LM/blob/0fecd76e995c136021d478c6c52caa57c2f9aa25/tools/retro/build_db.md>`_. The main stages of the process are summarized below. 

The output result of the preparation step is a processed RETRO data directory ready to be used for pre-training. Particularly, it  will contain the main following files and subdirectories:

* ``config.json``: containing the hyperparameters used in the data preparation step, which will be retrieved to use in the pre-training step for consistency. For example: sample length, chunk length, data splits, tokenizer files, etc.
* ``data``: containing the original data before any preprocessing.
* ``tokenizer``: containing tokenizer files used in the preparation step.
* ``db``: containing the chunk database of processed and chunked text used for retrieving neighbors. 
* ``index``: containing the Faiss index of the chunk database for retrieval.
* ``query``: containing the queried neighboring chunks for all training samples.

Summary of Main Stages
%%%%%%%%%%%%%

The data preparation process contains the following main stages:

Build Retrieval Chunk Database
##############################

This stage involves creating a database of text chunks from a corpus such as Wikipedia to be used for retrievals. The chunks are non-overlapping and extracted from the original GPT token dataset, with each chunk traditionally being 64 tokens in length. The database is stored as a 2-D array and is not a relational database. 

The main output of this stage are:

* ``/db/merged/train.hdf5``: the database containing all processed and chunked text.
* ``/db/merged/sampled.hdf5``: the database containing a small portion of all chunks, only used for training the index in the next stage.

Build Index for Similarity Search
#################################

The second stage is to build a search index using Faiss, a library for efficient similarity search. The index is trained on a subset of the chunks ``sampled.hdf5`` from the database. After training, all chunks are added to the index to enable querying. The index accepts 1-D floating point vectors, so chunks must be embedded using Bert embeddings before they can be added to the index. Particularly, the stage is comprised of two sub-stages:

    \- Extract BERT embeddings from the sampled chunk database (``sampled.hdf5``) and use them to train a Faiss index.
    
    \- Extract BERT embeddings for each chunk in the all chunks database (``train.hdf5``) and add them to the trained Faiss index.

The main output of this stage are:

* ``/index/<RETRO_INDEX_TYPE>/<RETRO_INDEX_STR>/added.faissindex``: the trained index, with all chunks in the database added to it

Query Pretraining Neighbors
###########################

For training RETRO, to speed up the Retro pretraining process, we will pre-retrieve neighbors for all training samples instead of retrieving them on-the-fly. In this stage, the pretraining datasets are processed to find and save k-nearest neighbors for each chunk in each sample. The neighbors are saved to disk and labeled with unique properties to ensure they match the pretraining configuration. Query-time hyperparameters can be tuned to improve the quality of the neighbors.

The main output of this stage are:

* ``train_<UNIQUE_HASH>``: directory containing retrieved neighbors for all training samples.
* ``valid_<UNIQUE_HASH>``: directory containing retrieved neighbors for all validating samples.

Train NeMo RETRO Model
-----------------------

Once the training data, retrieval data, KNN index, and Faiss index are prepared, we are ready to train the RETRO model.
The training process will use the output directory from the data preparation step. We set the path to this directory at the retro.retro_project_dir variable in the training configuration yaml file. Many of the data hyperparameters will be retrieved from the config.json file in this directory, including data splits, sequence length, chunk length, number of training and validating samples, tokenizer, etc.

The table below lists some of the common parameters that can be configured for model pre-training.

+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| **Parameter**                    | **Default** | **Description**                                                                        |
+==================================+=============+========================================================================================+
| model.micro_batch_size           | 4           | the micro batch size used for training                                                 |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.tensor_model_parallel_size | 1           | tensor model parallel size                                                             |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.encoder_seq_length         | 2048        | token sequence length                                                                  |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.chunk_size                 | 64          | the chunk size used to retrieve                                                        |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.enc_num_layers             | 4           | total number of encoder layers                                                         |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.dec_num_layers             | 6           | total number of decoder layers                                                         |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.enc_cross_attention        | [3]         | layer numbers for cross attention in encoder                                           |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.dec_cross_attention        | [3,4,5]     | layer numbers for chunked cross attention in decoder                                   |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.add_position_embedding     | FALSE       | whether to add the absolute position encoding                                          |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.hidden_size                | 768         | model hidden size                                                                      |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.ffn_hidden_size            | 3072        | model FFN hidden size. Usually 4 * hidden_size                                         |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.num_attention_heads        | 12          | number of attention heads                                                              |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.init_method_std            | 0.02        | standard deviation of the zero mean normal distribution used for weight initialization |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.hidden_dropout             | 0.1         | dropout probability for hidden state transformer                                       |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.attention_dropout          | 0.1         | dropout probability in the attention layer                                             |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+
| model.ffn_dropout                | 0           | dropout probability in the feed-forward layer                                          |
+----------------------------------+-------------+----------------------------------------------------------------------------------------+

An example RETRO pre-training script is:

.. code-block:: bash

        python /lustre/fsw/coreai_dlalgo_genai/huvu/codes/retro/huy_nemo/NeMo_retro/examples/nlp/language_modeling/megatron_retro_pretraining.py \
            trainer.num_nodes=1 \
            trainer.devices=8 \
            trainer.precision=bf16 \
            trainer.accelerator=gpu \
            model.data.data_prefix=["none"] \
            exp_manager.exp_dir=/lustre/fsw/coreai_dlalgo_genai/huvu/data/retro/mcore_retro_dataloader/nemo_cyclic_eos_wiki_ca5b3989 \
            exp_manager.create_wandb_logger=True \
            exp_manager.wandb_logger_kwargs.name=mcore_retro_testing_junks \
            exp_manager.wandb_logger_kwargs.project=mcore_retro_interactive \
            +exp_manager.wandb_logger_kwargs.resume=False \
            model.mcore_gpt=True \
            model.tensor_model_parallel_size=1 \
            model.pipeline_model_parallel_size=1 \
            model.optim.name=distributed_fused_adam \
            model.retro.retro_project_dir=/lustre/fsw/coreai_dlalgo_genai/huvu/data/retro/pretrain_data/wiki-core-bert-fast \
            model.data.num_workers=4 \
            ++cluster_type=BCP \
            model.micro_batch_size=4 \
            model.data.shuffle_documents=False \
            trainer.val_check_interval=10 \
            model.init_method_std=0.023 \
            model.optim.lr=6.0e-4 \
            model.optim.weight_decay=0.1 \
            model.optim.sched.name=CosineAnnealing \
            model.optim.sched.min_lr=6.0e-5 \
            model.optim.sched.max_steps=650000 \
            model.megatron_amp_O2=True \
            model.data.dataloader_type=cyclic \
            model.data.splits_string=\'98,2,0\' \
            trainer.max_steps=750000


During the training, launch Tensorboard to monitor training like so:

.. code-block:: bash

    tensorboard --logdir /result/retro_model --bind_all

.. note:: Weights and Biases (WandB) is supported too. Add ``exp_manager.create_wandb_logger=True`` to the model training arguments to enable it.

After the training, the model nemo file can be found at the result checkpoint directory.

Run NeMo RETRO Model Inference
-------------------------------

Once the NeMo RETRO model has been trained, we can put it into inference mode and experiment with it. 
During inference, we are not limited to the static Faiss index that we built earlier for KNN queries. 
We can feed any external data to the model as retrieval context. NeMo RETRO implementation supports dynamic retrieval service, 
allowing users to add, reset, and query new documents on the fly.

When inferencing, input for RETRO is set up differently than when in training. Particularly, the model's input will be presented as comprising of two chunks only, one for the prompt, and one for the answer to be generated. These chunks don't necessarily have the length of 64 as in training, but will have the length of the tokenized prompt. For each prompt, context neighbors can be provided. These neighbors will correspond to the first chunk and will be passed through RETRO's encoder to generate text for the second chunk.

.. code-block:: bash

        python /lustre/fsw/coreai_dlalgo_genai/huvu/codes/retro/huy_nemo/NeMo_retro_eval/examples/nlp/language_modeling/megatron_retro_eval.py \
            checkpoint_dir=/lustre/fsw/coreai_dlalgo_genai/huvu/data/retro/mcore_retro_dataloader/mcore_retro_mlmcheckpoint_converting/megatron_gpt/checkpoints \
            checkpoint_name=\'megatron_gpt--val_loss=2.36-step=2-consumed_samples=512.0-last\' \
            inference.greedy=False \
            inference.add_BOS=False \
            inference.tokens_to_generate=10 \
            inference.temperature=1.0 \
            trainer.devices=1 \
            trainer.num_nodes=1 \
            trainer.accelerator=gpu \
            trainer.precision=32 \
            megatron_amp_O2=False \
            ++cluster_type=BCP \
            tensor_model_parallel_size=-1 \
            pipeline_model_parallel_size=-1 \
            inference.retro_inference.retro_gpt_retrieved_length=128 \
            inference.retro_inference.retro_num_neighbors=3 \
            inference.retro_inference.ft_neighbours=0 \
            inference.retro_inference.reuse_top=False \
            prompt="Question: Who is the current president of the US in 2024? Answer:" \
            neighbors=["The president of the US in 2024 is Joe Biden","The president of the US in 2024 is Joe Biden","The president of the US in 2024 is Joe Biden"]

Set the retro_model_file to use the nemo file generated in the pre-training step. After launching the server, copy-paste the URL from 
the terminal into your browser. Use the specified username and password to log in and have fun experimenting with the RETRO model.

References
************

.. bibliography:: ../../nlp_all.bib
    :style: plain
    :labelprefix: nlp-retro
    :keyprefix: nlp-retro-
