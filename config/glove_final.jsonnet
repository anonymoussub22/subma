local bert_dim = 300+128;
local epochs = 50;
local learn_rate = 0.001;
local weight_decay = 1.73e-6;
local dropout = 0.05;
{
    dataset_reader : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        type: "pre-parsed",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
        token_indexers: {
            tokens: {
                type: "single_id",
            },
            token_characters: {
                type: "characters",
                min_padding_length: 3
            },
        },
    },

    train_data_path: "./data/semeval/split_train.txt",
    validation_data_path: "./data/semeval/split_val.txt",

    model: {
        type: "graph_conv",
        embedder: {
            token_embedders: {
                tokens: {
                    type: "embedding",
                    embedding_dim: 300,
                    pretrained_file: "../vectors/glove.840B.300d.txt",
                    trainable: true,
               },
               token_characters: {
                    type: "character_encoding",
                    embedding: {
                        embedding_dim: 128,
                        vocab_namespace: "token_characters"
                    },
                    encoder: {
                        type: "cnn",
                        embedding_dim: 128,
                        num_filters: 128,
                        ngram_filter_sizes: [
                            3
                        ],
                        conv_layer_activation: "gelu"
                    }
                },
            },

        },
        encoder:{
            type: "gru",
            input_size: bert_dim,
            hidden_size: bert_dim/2,
            num_layers: 3,
            dropout: dropout,
            bidirectional: true,
        },
        pooler: {
            type: "gru",
            input_size: bert_dim,
            hidden_size: bert_dim/2,
            num_layers: 1,
            bidirectional: true,
        },
        hidden_size: bert_dim,
        hidden_dropout_prob: dropout,
        initializer: {
            regexes: [
                 [
                    "ensemble_linear.weight",
                    {
                        "type": "xavier_uniform"
                    },
                ],
                [
                    "entity_extractor._global_attention._module.weight",
                    {
                        "type": "xavier_uniform"
                    },
                ],
                [
                    "classifier.weight",
                    {
                        "type": "xavier_uniform"
                    },
                ],
                [
                    "dep_type_embedding.weight",
                    {
                        "type": "xavier_uniform"
                    },
                ],
                [
                    "pooler.pooler.dense.weight",
                    {
                        "type": "xavier_uniform"
                    },
                ],
            ]
        },
    },

    data_loader: {
        batch_sampler:{
            batch_size: 32,
            type: 'bucket',
        },
    },
    trainer: {
        num_epochs: epochs,
        learning_rate_scheduler:{
            type: "linear_with_warmup",
            warmup_steps: 100,
        },
        patience: 10,
        cuda_device: 0,
        validation_metric: "+fscore",
        optimizer: {
            type: "adam",
            lr:learn_rate,
            weight_decay:weight_decay,
        },
    }
}
