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


Train NeMo RETRO Model
-----------------------

Once the training data, retrieval data, KNN index, and Faiss index are prepared, we are ready to train the RETRO model. In the NeMo implementation, 
the RETRO model can be pre-trained with or without the `mu-Transfer <https://openreview.net/pdf?id=Bx6qKuBM2AD>`_ :cite:`nlp-retro-yang2022tensor` feature. We will introduce both ways.


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

    python examples/nlp/language_modeling/megatron_retro_pretraining.py \
        trainer.devices=8 \
        trainer.num_nodes=2 \
        trainer.accelerator=gpu \
        trainer.max_steps=800000 \
        trainer.precision=16 \
        exp_manager.exp_dir=/result/retro_model \
        model.apply_query_key_layer_scaling=False \
        model.tensor_model_parallel_size=8 \
        model.optim.name=adamw \
        model.enc_num_layers=2 \
        model.dec_num_layers=32 \
        model.enc_cross_attention=[0] \
        model.dec_cross_attention=[8,11,14,17,20,23,26,29,31] \
        model.hidden_size=4096 \
        model.ffn_hidden_size=16384 \
        model.num_attention_heads=32 \
        model.tokenizer.merge_file=/dataset/gpt2-merges.txt \
        model.tokenizer.vocab_file=/dataset/gpt2-vocab.json \
        model.data.data_prefix=[/result/pubmed_eval_text_document] \
        model.data.knn_index=[dataset/pubmed_knn_final.save] \
        model.data.retrieval_prefix=/result/pubmed_eval_text_document \
        model.micro_batch_size=8

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

We have built a simple web client that makes it easy for users to play around with the model. Here is an example script to launch the server:

.. code-block:: bash

    python examples/nlp/language_modeling/megatron_retro_eval.py \
        trainer.devices=8 \
        trainer.num_nodes=1 \
        trainer.accelerator=gpu \
        trainer.precision=16 \
        retro_model_file=megatron_retro.nemo \
        tensor_model_parallel_size=8 \
        pipeline_model_parallel_size=1 \
        retrieval_service.sentence_bert.devices=\'0,1,2,3,4,5,6,7\' \
        retrieval_service.services.0.faiss_devices=\'0,1,2,3,4,5,6,7\' \
        retrieval_service.services.1.faiss_devices=\'0,1,2,3,4,5,6,7\' \
        retrieval_service.services.0.faiss_index=/result/pubmed_faiss_final.index \
        retrieval_service.services.0.retrieval_index=/result/pubmed_eval_text_document \
        retrieval_service.neighbors=2 \
        retrieval_service.pad_tokens=True \
        retrieval_service.store_retrieved=True \
        server=True \
        web_server=True \
        share=True \
        username=test \
        password=test123

Set the retro_model_file to use the nemo file generated in the pre-training step. After launching the server, copy-paste the URL from 
the terminal into your browser. Use the specified username and password to log in and have fun experimenting with the RETRO model.

References
************

.. bibliography:: ../../nlp_all.bib
    :style: plain
    :labelprefix: nlp-retro
    :keyprefix: nlp-retro-
