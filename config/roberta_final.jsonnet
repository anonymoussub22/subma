local bert_model = "roberta-base";
local bert_dim = 768;

local epochs = 50;

local learn_rate = 1.5e-5;
local weight_decay = 1.65e-5;
local dropout = 0.02;

{
    dataset_reader : {
        // This name needs to match the name that you used to register your dataset reader, with
        // the call to `@DatasetReader.register()`.
        type: "pre-parsed",
        // These other parameters exactly match the constructor parameters of your dataset reader class.
        token_indexers: {
            tokens: {
                type: "pretrained_transformer_mismatched",
                model_name: bert_model,
            },
        }
    },

    train_data_path: "./data/semeval/split_train.txt",
    validation_data_path: "./data/semeval/split_val.txt",

    model: {
        type: "graph_conv",
        embedder: {
            token_embedders: {
                tokens: {
                    type: "pretrained_transformer_mismatched",
                    model_name: bert_model,
                },
            }
        },
        pooler: {
            type: "bert_pooler",
            pretrained_model: bert_model,
        },
        hidden_size: bert_dim,
        hidden_dropout_prob: dropout,
        initializer: {
            regexes: [
                [
                    "classifier.weight",
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
