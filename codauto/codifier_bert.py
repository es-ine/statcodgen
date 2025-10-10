# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (C) [2025] Instituto Nacional de Estadística
#
# Este archivo forma parte del proyecto statcodgen.
#
# Licenciado bajo la Licencia Pública de la Unión Europea (EUPL) v.1.2.
# Puede obtener una copia de la licencia en la raiz de este proyecto o en:
# https://eupl.eu/1.2/es/
#
# A menos que se indique lo contrario, este software se distribuye
# "TAL CUAL", SIN GARANTÍAS NI CONDICIONES DE NINGÚN TIPO.
# Consulte la licencia para conocer los términos específicos.
# ------------------------------------------------------------------------------
# Copyright (C) [2025] National Institute of Statistics
#
# This file is part of the statcodgen project.
#
# Licensed under the European Union Public License (EUPL) v.1.2.
# You can obtain a copy of the license at the root of this project or at:
# https://eupl.eu/1.2/es/
#
# Unless otherwise indicated, this software is distributed
# "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the license for specific terms.
# ------------------------------------------------------------------------------
"""
Created on Tue Sep  9 13:40:12 2025

@author: U853768
"""
import numpy as np
import torch
import random
import datetime
import time
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from codauto.codifier import Codifier


class CodifierBERT(Codifier):
    """
    Codifier implementation based on a pre-trained Spanish BERT model
    (dccuchile/bert-base-spanish-wwm-uncased) for sequence classification.

    This class extends the abstract `Codifier` by providing concrete methods
    to tokenize text data, prepare datasets, fine-tune BERT for hierarchical
    classification tasks, and generate predictions with confidence scores.

    Attributes
    ----------
    tokenizer : transformers.BertTokenizer
        Tokenizer associated with the pre-trained BERT model.
    training_stats : list
        A list of dictionaries storing training statistics (loss, accuracy,
        training time, validation time, etc.) across epochs.

    Methods
    -------
    flat_accuracy(preds, labels):
        Compute accuracy from predicted logits and ground-truth labels.
    format_time(elapsed):
        Convert elapsed seconds into a human-readable string (h:mm:ss).
    index_to_label():
        Create bidirectional mappings between label strings and integer indices
        from the training dataset.
    max_len():
        Compute the maximum tokenized sequence length across the training set.
    tokenize_dataset():
        Encode the training dataset into BERT-compatible tensors
        (input IDs, attention masks, and labels).
    split_dataset():
        Split the encoded dataset into training and validation subsets (90/10).
    dataloader(batch_size):
        Build DataLoader objects for both training and validation sets.
    optimizer(trainable_layers, learning_rate, weight_decay, batch_size, epochs):
        Initialize a BERT classification model, configure which encoder layers
        to unfreeze, and set up optimizer and learning-rate scheduler.
    train(**kwargs):
        Fine-tune the BERT model on the training dataset for the specified
        number of epochs. Tracks loss and accuracy on both training and
        validation sets.
    save(name):
        Save the fine-tuned model’s state dictionary to disk.
    load(name):
        Load a previously saved model from disk and prepare it for inference.
    get_pred_for_batch(samples, idxs):
        Generate predictions for a batch of text samples, returning labels
        and their associated confidence scores.
    """

    def __init__(self,
                 structure_instance,
                 train_df,
                 test_df,
                 root_path,
                 corres_df=None,
                 min_lenght_texts=3
                 ):
        super().__init__(
            structure_instance=structure_instance,
            train_df=train_df,
            test_df=test_df,
            root_path=root_path,
            corres_df=corres_df,
            min_lenght_texts=min_lenght_texts
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-uncased'
        )
        self.training_stats = []

    @staticmethod
    def flat_accuracy(preds, labels):
        """
       Compute the accuracy of predictions.

       Parameters
       ----------
       preds : numpy.ndarray
           Array of model logits or probabilities with shape
           (n_samples, n_classes).
       labels : numpy.ndarray
           Array of true class indices with shape (n_samples,).

       Returns
       -------
       float
           The fraction of correctly predicted labels.
       """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @staticmethod
    def format_time(elapsed):
        """
        Format elapsed time into a human-readable string.

        Parameters
        ----------
        elapsed : float
            Time in seconds.

        Returns
        -------
        str
            Formatted time in h:mm:ss format.
        """
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def index_to_label(self):
        """
        Create mappings between labels and indices.

        Returns
        -------
        tuple of dict
            - index_to_label : dict
                Mapping from integer indices to label strings.
            - label_to_index : dict
                Mapping from label strings to integer indices.

        Notes
        -----
        Labels are extracted from the first column of `train_df` and sorted.
        """
        unique_labels = sorted(set(self.train_df.iloc[:, 0]))
        label_to_index = {
            label: idx
            for idx, label in enumerate(unique_labels)
        }
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        return index_to_label, label_to_index

    def max_len(self):
        """
        Compute the maximum tokenized length of the training dataset.

        Returns
        -------
        int
            Maximum sequence length (number of tokens) after BERT tokenization.

        Notes
        -----
        - Uses `self.tokenizer.encode` with special tokens and truncation
          enabled.
        - Useful to set the maximum sequence length for padding and batching.
        """
        return max(
            len(self.tokenizer.encode(
                text, add_special_tokens=True, truncation=True
            )) for text in self.train_df.iloc[:, 1]
        )

    def tokenize_dataset(self):
        """
        Tokenize the training dataset into BERT-compatible tensors.

        Returns
        -------
        tuple of torch.Tensor
            - input_ids : torch.Tensor
                Token IDs for all training samples
                (shape: [n_samples, max_len]).
            - attention_mask : torch.Tensor
                Attention masks indicating padded tokens
                (shape: [n_samples, max_len]).
            - labels : torch.Tensor
                Ground-truth labels as integer indices (shape: [n_samples]).

        Notes
        -----
        - Pads all sequences to the maximum length found in the dataset.
        - Encodes text from the second column of `train_df`.
        - Converts labels from the first column into indices using
          `index_to_label()`.
        """
        max_length = self.max_len()
        dataset_text = self.train_df.iloc[:, 1]
        input_ids = []
        attention_mask = []
        for text in dataset_text:
            encoded_dict = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_mask.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        _, label_to_index = self.index_to_label()
        labels = torch.tensor([
            label_to_index[label]
            for label in self.train_df.iloc[:, 0]
        ], dtype=torch.long)

        return input_ids, attention_mask, labels

    def split_dataset(self):
        """
        Split the tokenized dataset into training and validation subsets.

        Returns
        -------
        tuple of torch.utils.data.Dataset
            - train_dataset : TensorDataset
                90% of the samples for training.
            - val_dataset : TensorDataset
                10% of the samples for validation.

        Notes
        -----
        The split ratio is fixed at 90/10.
        """
        input_ids, attention_mask, labels = self.tokenize_dataset()
        dataset = TensorDataset(input_ids, attention_mask, labels)
        train_size = int(0.9*len(dataset))
        val_size = len(dataset)-train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        return train_dataset, val_dataset

    def dataloader(self, batch_size):
        """
        Create DataLoader objects for training and validation sets.

        Parameters
        ----------
        batch_size : int
            Number of samples per batch.

        Returns
        -------
        tuple of torch.utils.data.DataLoader
            - train_dataloader : DataLoader
                DataLoader for the training subset (randomly sampled).
            - validation_dataloader : DataLoader
                DataLoader for the validation subset (sequentially sampled).
        """
        train_dataset, val_dataset = self.split_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size
        )
        validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size
        )
        return train_dataloader, validation_dataloader

    def optimizer(
        self, trainable_layers, learning_rate, weight_decay, batch_size, epochs
    ):
        """
        Initialize BERT, configure trainable layers, and set up
        optimizer/scheduler.

        Parameters
        ----------
        trainable_layers : list of int
            Indices of BERT encoder layers to unfreeze for fine-tuning.
            Negative values are supported (e.g., -1 for the last layer).
        learning_rate : float
            Initial learning rate for the optimizer.
        weight_decay : float
            Weight decay (L2 regularization).
        batch_size : int
            Batch size for training.
        epochs : int
            Number of training epochs (used to compute scheduler steps).

        Returns
        -------
        tuple
            - model : BertForSequenceClassification
                BERT classification model initialized with the correct number
                of labels.
            - optimizer : torch.optim.Optimizer
                AdamW optimizer for the unfrozen parameters.
            - scheduler : torch.optim.lr_scheduler.LambdaLR
                Linear learning rate scheduler with warmup.

        Notes
        -----
        - All encoder layers are frozen by default, then selected layers are
          unfrozen.
        - If `trainable_layers` is empty, the encoder remains frozen.
        """
        model = BertForSequenceClassification.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-uncased',
            num_labels=len(set(self.train_df.iloc[:, 0])),
            output_attentions=False,
            output_hidden_states=False
        )

        num_layers = len(model.bert.encoder.layer)
        valid_layers = [
            i if i >= 0 else num_layers + i for i in trainable_layers
        ]

        if not valid_layers:
            self.logger.info(
                "No layers in the BERT model were unfrozen for training"
            )

        valid_layers = [i for i in valid_layers if 0 <= i < num_layers]

        for param in model.bert.parameters():
            param.requires_grad = False

        for layer_idx in valid_layers:
            for param in model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = True

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            eps=1e-8,
            weight_decay=weight_decay
        )
        total_steps = len(self.split_dataset()[0]) // batch_size * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        return model, optimizer, scheduler

    def train(self, **kwargs):
        """
        Fine-tune the BERT model on the training dataset.

        Parameters
        ----------
        **kwargs : dict, optional
            Training configuration. Supported keys:
            - 'epoch' (int): number of epochs (default=5).
            - 'learning_rate' (float): learning rate (default=3e-5).
            - 'batch_size' (int): batch size (default=16).
            - 'weight_decay' (float): weight decay (default=0.01).
            - 'trainable_layers' (list of int): encoder layers to unfreeze
              (default=[-1]).

        Returns
        -------
        None

        Notes
        -----
        - Trains the model for the given number of epochs.
        - Logs training/validation loss and accuracy at each epoch.
        - Stores training statistics in `self.training_stats`.
        - Saves the trained model in `self.model`.
        """
        epochs = kwargs.get('epoch', 5)
        learning_rate = kwargs.get('learning_rate', 3e-5)
        batch_size = kwargs.get('batch_size', 16)
        weight_decay = kwargs.get('weight_decay', 0.01)
        trainable_layers = kwargs.get('weight_decay', [-1])

        train_dataloader, validation_dataloader = self.dataloader(batch_size)
        model, optimizer, scheduler = self.optimizer(
            trainable_layers, learning_rate, weight_decay, batch_size, epochs
        )
        num_labels = len(set(self.train_df.iloc[:, 0]))

        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        training_stats = []
        total_t0 = time.time()

        for epoch_i in range(epochs):
            self.logger.info(
                f'\n======== Epoch {epoch_i+1} / {epochs} ========\nTraining')
            t0 = time.time()
            total_train_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time()-t0)
                    message_1 = '   Batch{:>5,}  of  {:>5,}.    Elapsed:{:}.'.format(
                        step, len(train_dataloader), elapsed
                    )
                    self.logger.info(message_1)

                b_inputs_ids = batch[0]
                b_input_mask = batch[1]
                b_labels = batch[2]
                model.zero_grad()

                # forward
                logits = model(
                    b_inputs_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask
                )[0]
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), b_labels.view(-1))
                total_train_loss += loss.item()

                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = self.format_time(time.time()-t0)
            message_1 = "  Average training loss: {0:.2f}".format(
                avg_train_loss
            )
            message_2 = "  Training time: {0}".format(training_time)
            self.logger.info(f'\n{message_1}\n{message_2}\nRunning Validation')
            t0 = time.time()
            model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0

            for batch in validation_dataloader:
                b_input_ids = batch[0]
                b_input_mask = batch[1]
                b_labels = batch[2]
                with torch.no_grad():
                    logits = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask
                    )[0]
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        logits.view(-1, num_labels),
                        b_labels.view(-1)
                    )
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.numpy()
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            message_1 = "  Accuracy: {0:.2f}".format(avg_val_accuracy)
            self.logger.info(message_1)
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            validation_time = self.format_time(time.time() - t0)
            message_1 = "  Validation Loss: {0:.2f}".format(avg_val_loss)
            message_2 = "  Validation time: {0}".format(validation_time)
            self.logger.info(f'\n{message_1}\n{message_2}')
            training_stats.append({
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
        self.training_stats.append(training_stats)
        message_1 = "Total training took {0} (h:mm:ss)".format(
            self.format_time(time.time()-total_t0)
        )
        self.logger.info(f'\nTraining complete\n{message_1}')
        self.model = model

    def save(self, name):
        """
        Save the model's state dictionary to disk.

        This method stores only the model parameters (weights and biases), not
        the full model architecture. To reload the model, the same architecture
        must be instantiated and the state dictionary loaded into it.

        Parameters
        ----------
        name : str
            File name (without extension). The state dictionary is saved as
            '<name>.pth' in `self.root_path`.

        Returns
        -------
        None
        """
        model_path = os.path.join(self.root_path, name+'.pth')
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f'Model saved as: {model_path}')

    def load(self, name):
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        name : str
            File name (without extension). The method expects '<name>.pth'
            to be located in `self.root_path`.

        Returns
        -------
        None

        Notes
        -----
        The model is reinitialized with the correct label space before
        loading the state dictionary.
        """
        load_path = os.path.join(self.root_path, name+'.pth')
        model = BertForSequenceClassification.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-uncased',
            num_labels=len(set(self.train_df.iloc[:, 0])),
            output_attentions=False,
            output_hidden_states=False
        )
        model.load_state_dict(torch.load(
            load_path, map_location=torch.device('cpu')
        ))
        model.eval()
        self.model = model

    def get_pred_for_batch(self, samples):
        """
        Generate predictions for a batch of text samples.

        Parameters
        ----------
        samples : list of str
            Input text samples to classify.

        Returns
        -------
        list of tuple
            A list where each element corresponds to one input sample and
            contains:
            - labels (list of str): predicted labels, sorted by confidence.
            - confidences (list of float): corresponding probabilities.

        Notes
        -----
        - Applies softmax to model logits to obtain probabilities.
        - Uses `torch.topk` to retrieve the top-N predictions, where
          N equals the number of classes in the hierarchy.
        """
        index_to_label, label_to_index = self.index_to_label()
        predictions = [None] * len(samples)
        n_classes = len(self.structure.reversed_hierarchy.keys())
        for idx, sample in enumerate(samples):
            inputs = self.tokenizer(sample, padding=True, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            top_probs, top_classes = torch.topk(probs, n_classes, dim=1)
            top_class = []
            for topk_probs_row in top_classes.tolist():
                topk_classes_row = [
                    index_to_label[label]
                    for label in topk_probs_row
                ]
                top_class.append(topk_classes_row)
            predictions[idx] = (top_class[0], top_probs[0].tolist())
        return predictions
