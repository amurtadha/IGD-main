import operator
import logging
import argparse
import os
import sys
from time import strftime, localtime
import random
import numpy
from transformers import AdamW
import  copy
from sklearn import metrics
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_utils import   custom_dataset, custom_dataset_pl
from tqdm import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
import json
from transformers import  AutoTokenizer
from MyModel import FineTuning,  MyIGD
from captum.attr import IntegratedGradients
from captum.attr import visualization
from collections import defaultdict



class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.stop_words={'ourselves', 'hers', 'between', 'yourself',  'agnewsain', 'there', 'about', 'once', 'during', 'out', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through',  'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all',  'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did',  'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'agnewsainst', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
        self.tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_bert_name)

        self.opt.labels = json.load(open('datasets/{0}/labels.json'.format(opt.dataset)))
        self.opt.lebel_dim = len(self.opt.labels)

        self.testset = custom_dataset(opt.dataset_file['test'], self.tokenizer, opt.max_seq_len,self.opt.labels)
        self.valset = custom_dataset(opt.dataset_file['dev'], self.tokenizer, opt.max_seq_len, self.opt.labels)
        self.trainset = custom_dataset(opt.dataset_file['train'], self.tokenizer, opt.max_seq_len, self.opt.labels)
        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        self.counter = json.load(open('datasets/counter/counterfitted_neighbors.json'))
        self.ig = IntegratedGradients(self.forward_model)
        self.vis_data_records_ig = []
        logger.info('{} {} {}'.format(len(self.trainset), len(self.testset), len(self.valset)))


        self.model = MyIGD(opt)

        if opt.parrallel:
            self.model = nn.DataParallel(self.model)

        self.model.to(opt.device)
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))

    def forward_model(self, inputs=None):
        outputs = self.compute_bert_outputs(inputs)
        pooled_output = outputs[1]
        try:
            pooled_output = self.model_generator.encoder.dropout(pooled_output)
        except:
            pass
        logits = self.model_generator.encoder.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)

    def compute_bert_outputs(self, embedding_output, attention_mask=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if self.opt.parrallel:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.module.encoder.parameters()).dtype)  # fp16 compatibility
        else:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.encoder.parameters()).dtype)  # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                if self.opt.parrallel:
                    head_mask = head_mask.expand(self.model.encoder.config.num_hidden_layers, -1, -1, -1, -1)
                else:
                    head_mask = head_mask.expand(self.model.module.encoder.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.model.encoder.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            if self.opt.parrallel:
                head_mask = [None] * self.model_generator.module.encoder.config.num_hidden_layers
            else:
                head_mask = [None] * self.model_generator.encoder.config.num_hidden_layers
        if self.opt.parrallel:
            encoder_outputs = self.model_generator.module.encoder.encoder(embedding_output,
                                                              extended_attention_mask,
                                                              head_mask=head_mask)
            sequence_output = encoder_outputs[0]
            pooled_output = self.model_generator.module.encoder.pooler(sequence_output)

        else:
            
            if self.opt.plm=='bert':
                encoder_outputs=  self.model_generator.encoder.bert.encoder(embedding_output,
                                                             extended_attention_mask,
                                                             head_mask=head_mask)
                sequence_output = encoder_outputs[0]
                pooled_output = self.model_generator.encoder.bert.pooler(sequence_output)
            else:
                encoder_outputs = self.model_generator.encoder.roberta.encoder(embedding_output,
                                                                               extended_attention_mask,
                                                                               head_mask=head_mask)
                sequence_output = encoder_outputs[0]
                pooled_output= sequence_output
                # pooled_output = self.model_generator.encoder.roberta.pooler(sequence_output)

        # pooled_output = self.model.encoder.bert.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def add_attributions_to_visualizer_batch(self, attributions, tokens, pred, pred_ind, label, delta):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        # attributions=attributions[1:-1]
        attributions=attributions[:,1:-1]

        for i in range(attributions[:,1:-1].shape[0]):
            self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
                attributions[i],
                pred[i].item(),
                self.Label[pred_ind[i]],
                self.Label[label[i].item()],
                "label",
                attributions.sum(),
                tokens[i][1:len(attributions[i])],
                delta[i].item()))

        # storing couple samples in an array for visualization purposes
        # self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
        #     attributions,
        #     pred,
        #     self.Label[pred_ind],
        #     self.Label[label],
        #     "label",
        #     attributions.sum(),
        #     tokens[1:len(attributions)],
        #     delta))
    def add_attributions_to_visualizer(self, attributions, tokens, pred, pred_ind, label, delta):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        attributions=attributions[1:-1]


        # storing couple samples in an array for visualization purposes
        self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
            attributions,
            pred,
            self.Label[pred_ind],
            self.Label[label],
            "label",
            attributions.sum(),
            tokens[1:len(attributions)+1],
            delta))

    def interpret_sentence_individual(self, inputs, label=1):
        input_ids = torch.tensor([self.tokenizer.encode(inputs, add_special_tokens=True)]).to(self.opt.device)
        # input_ids, token_type_ids, attention_mask, label = inputs
        self.model_generator.eval()
        self.model_generator.zero_grad()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens=tokens[:tokens.index(self.tokenizer.sep_token)+1]
        input_embedding = self.model_generator.encoder.bert.embeddings(input_ids[:len(tokens)])

        # predict
        pred = self.forward_model(input_embedding).item()
        pred_ind = round(pred)

        # compute attributions and approximation delta using integrated gradients
        # reference_indices = [self.tokenizer.pad_token_id] * self.opt.max_seq_len
        # reference_indices[0], reference_indices[
        #     tokens.index(self.tokenizer.sep_token)] = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        # reference_indices = torch.tensor([reference_indices], device=self.opt.device)

        attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)

        # print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))
        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        attributions = attributions[1:-1]
        tokens =  tokens[1:len(attributions) + 1]
        logger.info(tokens)
        logger.info(attributions)
        logger.info('''''''''''''''''')
        # storing couple samples in an array for visualization purposes
        self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
            attributions,
            pred,
            self.Label[pred_ind],
            self.Label[label],
            "label",
            attributions.sum(),
            tokens,
            delta))


    def interpret_sentence(self, inputs, label=1):
        # input_ids = torch.tensor(self.tokenizer.encode(inputs, add_special_tokens=True)).to(self.opt.device)
        input_ids, token_type_ids, attention_mask, label = inputs
        self.model_generator.eval()
        self.model_generator.zero_grad()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        tokens=tokens[:tokens.index(self.tokenizer.sep_token)+1]
        input_embedding = self.model_generator.encoder.bert.embeddings(input_ids[:len(tokens)].unsqueeze(0))

        # predict
        pred = self.forward_model(input_embedding).item()
        pred_ind = round(pred)

        # compute attributions and approximation delta using integrated gradients
        # reference_indices = [self.tokenizer.pad_token_id] * self.opt.max_seq_len
        # reference_indices[0], reference_indices[
        #     tokens.index(self.tokenizer.sep_token)] = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        # reference_indices = torch.tensor([reference_indices], device=self.opt.device)

        attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)

        # print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))


        self.add_attributions_to_visualizer(attributions_ig, tokens, pred, pred_ind, label.item(), delta)
    def interpret_sentence_experiment(self, inputs):
        input_ids, token_type_ids, attention_mask, label = inputs
        self.model_generator.eval()
        self.model_generator.zero_grad()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        tokens=tokens[:tokens.index(self.tokenizer.sep_token)+1]
        # try:
        input_embedding = self.model_generator.encoder.bert.embeddings(input_ids[:len(tokens)].unsqueeze(0))
        # except:
        #     input_embedding= self.model_generator.encoder.embeddings(input_ids[:len(tokens)].unsqueeze(0))

        # predict
        # pred = self.forward_model(input_embedding).item()
        # pred_ind = round(pred)
        # attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)
        attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=100, return_convergence_delta=True)
        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        attributions = attributions[1:-1]

        return tokens[1:len(attributions) + 1], attributions, label.item()
    def interpret_sentence_experiment_batch(self, inputs):
        input_ids, token_type_ids, attention_mask, label = inputs
        self.model_generator.eval()
        self.model_generator.zero_grad()
        tokens = np.asarray([np.asarray(self.tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True)) for t in input_ids])
        if self.opt.plm =='bert':
            input_embedding = self.model_generator.encoder.bert.embeddings(input_ids)
        elif self.opt.plm == 'roberta':
            input_embedding = self.model_generator.encoder.roberta.embeddings(input_ids)

        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # tokens=tokens[:tokens.index(self.tokenizer.sep_token)+1]
        # # try:
        # input_embedding = self.model_generator.encoder.bert.embeddings(input_ids[:len(tokens)].unsqueeze(0))
        # except:
        #     input_embedding= self.model_generator.encoder.embeddings(input_ids[:len(tokens)].unsqueeze(0))

        # predict
        # pred = self.forward_model(input_embedding).item()
        # pred_ind = round(pred)
        # attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)
        attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=self.opt.n_steps_interpret, return_convergence_delta=True)
        attributions = attributions_ig.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        attributions =attributions[:,1:-1]


        # return tokens[1:len(attributions) + 1], attributions, label.item()
        return tokens, attributions, label


    def interpret_sentence_batch(self, inputs):
        input_ids, token_type_ids, attention_mask, labels = inputs
        self.model.eval()
        self.model.zero_grad()


        # input_ids = torch.tensor([self.tokenizer.encode(s, add_special_tokens=True) for s in sentences] ).to(self.opt.device)
        # input_embedding = self.model.bert.embeddings(input_ids)
        input_embedding = self.model.encoder.bert.embeddings(input_ids)

        # predict
        pred= self.forward_model(input_embedding)
        pred_ind =[round(p.item()) for p in pred]

        # compute attributions and approximation delta using integrated gradients
        attributions_ig, delta = self.ig.attribute(input_embedding, n_steps=30, return_convergence_delta=True)

        # print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))

        tokens = [self.tokenizer.convert_ids_to_tokens(d) for d in input_ids]
        self.add_attributions_to_visualizer_batch(attributions_ig, tokens, pred, pred_ind, labels, delta)

    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x
    def _train_warm(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        global_step = 0
        path = None

        for epoch in range(self.opt.num_epoch_warming):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0

            loss_total=[]
            self.model_generator.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):

                global_step += 1
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs_ori,_, _= self.model_generator(inputs[:3])

                targets= inputs[-1]
                loss_ori = criterion(outputs_ori, targets)
                loss = loss_ori
                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs_ori)
                    loss_total.append(loss.sum().detach().item())

            logger.info('epoch : {}'.format(epoch))
            logger.info('loss: {:.4f}'.format(np.mean(loss_total)))
            pres, recall, f1_score, acc = self._evaluate_acc_f1_warm(val_data_loader)
            logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f}'.format(pres, recall, f1_score, acc))
            if f1_score > max_val_acc:
                max_val_acc = f1_score
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = copy.deepcopy(self.model_generator.state_dict())

        return path

    def _train(self, criterion,optimizer, train_data_loader, val_data_loader, epoch_save=2):
        lamnd = self.opt.lamnd
        max_val_acc = 0
        global_step = 0
        path_save = '{}/{}_{}_final.bm'.format(self.opt.save_path, self.opt.dataset, self.opt.plm)
        if self.opt.inactivate_data_loss:
            path_save = '{}/{}_{}_final_wo_rt.bm'.format(self.opt.save_path, self.opt.dataset, self.opt.plm)


        len_dataloader = len(train_data_loader.dataset)
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0

            loss_total=[]
            loss_class=[]
            loss_src=[]
            self.model.train()
            for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
                # self.model.train()
                global_step += 1
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols_ads]
                targets = inputs[-1]

                domain_labels = torch.zeros_like(targets)
                try:
                    domain_labels[torch.tensor([i for i in range(0, int(inputs[0].shape[0] / 2), 2)])] = 1
                except:
                    continue
                if self.opt.inactivate_data_loss:
                    outputs_ori, _, _ = self.model(inputs, targets)
                    loss=loss_ori=loss_rev=criterion(outputs_ori, targets)
                else:

                    p = float(global_step + epoch * len_dataloader) / self.opt.num_epoch / len_dataloader
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    outputs_ori, _, outputs_ori_dom = self.model(inputs, label=targets, alpha=alpha)
                    loss_rev = criterion(outputs_ori_dom, domain_labels.to(self.opt.device))

                    loss_ori = criterion(outputs_ori, targets) # model training
                    loss = loss_ori+(loss_rev * lamnd)




                loss.sum().backward()

                optimizer.step()
                with torch.no_grad():
                    n_total += len(outputs_ori)
                    loss_total.append(loss.sum().detach().item())   
                    loss_class.append(loss_ori.sum().detach().item())
                    loss_src.append(loss_rev.sum().detach().item())

                if True and epoch>=epoch_save and  global_step % self.opt.log_step==0:
                    pres, recall, f1_score, acc = self._evaluate_acc_f1(val_data_loader)
                    logger.info(
                        '> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f}'.format(pres, recall,f1_score,acc))
                    self.model.train()
                    if f1_score > max_val_acc:
                        max_val_acc = f1_score
                        if not os.path.exists('state_dict'):
                            os.mkdir('state_dict')
                        path = copy.deepcopy(self.model.state_dict())
                        if self.opt.parrallel:
                            torch.save(self.model.module.state_dict(), path_save)
                        else:
                            torch.save(self.model.state_dict(), path_save)


            logger.info('epoch : {}'.format(epoch))
            logger.info('loss: {:.4f}, class : {:.4f},  src : {:.4f},  '.format(np.mean(loss_total),np.mean(loss_class),np.mean(loss_src)))
            pres, recall, f1_score, acc = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_precision: {:.4f}, val_recall: {:.4f}, val_f1: {:.4f},  val_acc: {:.4f}'.format(pres, recall, f1_score, acc))
            if epoch>=epoch_save and f1_score > max_val_acc:
                max_val_acc = f1_score
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')

                path = copy.deepcopy(self.model.state_dict())
                if self.opt.parrallel:
                    torch.save(self.model.module.state_dict(), path_save)
                else:
                    torch.save(self.model.state_dict(), path_save)

        return path


    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols_val]
                t_targets = t_inputs[-1]
                t_outputs,_,_ = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true= t_targets_all.cpu().detach().numpy().tolist()
            pred =torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()
            f= metrics.f1_score(true,pred,average='macro')
            r= metrics.recall_score(true,pred,average='macro')
            p= metrics.precision_score(true,pred,average='macro')
            acc= metrics.accuracy_score(true,pred)

        return  p, r, f, acc
    def _evaluate_acc_f1_warm(self,data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model_generator.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(tqdm(data_loader)):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols_val]
                t_targets = t_inputs[-1]
                t_outputs,_,_ = self.model_generator(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().detach().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets.detach()
                    t_outputs_all = t_outputs.detach()
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets.detach()), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.detach()), dim=0)
            true= t_targets_all.cpu().detach().numpy().tolist()
            pred =torch.argmax(t_outputs_all, -1).cpu().detach().numpy().tolist()
            f= metrics.f1_score(true,pred,average='macro')
            r= metrics.recall_score(true,pred,average='macro')
            p= metrics.precision_score(true,pred,average='macro')
            # classification_repo= metrics.classification_report(true, pred, target_names=list(labels.keys()))
            acc= metrics.accuracy_score(true,pred)

        return  p, r, f, acc





    def _generate_pseud_label(self, criterion, train_data_loader, val_data_loader):
        self.model_generator = FineTuning(self.opt)

        self.model_generator.to(self.opt.device)
        logger.info('train a generator')
        _params = filter(lambda p: p.requires_grad, self.model_generator.parameters())
        optimizer = self.opt.optimizer(self.model_generator.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        path = '{}/{}_{}_warm.bm'.format(self.opt.save_path, self.opt.dataset, self.opt.plm)
        if not os.path.exists(path):
            best_model_path = self._train_warm( criterion, optimizer, train_data_loader, val_data_loader)
            self.model_generator.load_state_dict(best_model_path)
            torch.save(self.model_generator.state_dict(), path)

        self.model_generator.load_state_dict(torch.load(path))
        if False:
            self.visualize_example(train_data_loader)
        data_to_save=[]
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size_generate, shuffle=True)
        for i_batch, sample_batched in enumerate(tqdm(train_data_loader)):
            inputs_batch = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            try:
                all_sentences_, all_scores_, label = self.interpret_sentence_experiment_batch(inputs_batch)
            except:continue
            all_labels_ = sample_batched['label']
            for i in range(len(sample_batched['input_ids'])):
                sentence_= all_sentences_[i]
                scores_= all_scores_[i]
                label= all_labels_[i].item()
                scores=[]
                sentence=[]
                j=0
                if self.opt.plm=='bert':
                    while j < len(sentence_):
                        word = sentence_[j]
                        score = scores_[j]
                        while j < len(sentence_)-1 and '##' in sentence_[j+1]:
                            j+=1
                            word+=sentence_[j].replace('##','')
                            score+=scores_[j]
                        j+=1
                        scores.append(score)
                        sentence.append(word)
                else:
                    while j < len(sentence_):
                        word = sentence_[j].replace('Ġ', '')
                        score = scores_[j]
                        while j < len(sentence_) - 1 and not sentence_[j + 1].startswith('Ġ'):
                            j += 1
                            word += sentence_[j].replace('##', '')
                            score += scores_[j]
                        j += 1
                        scores.append(score)
                        sentence.append(word)

                sentence=np.asarray(sentence)
                scores=np.asarray(scores)

                mn = np.mean(scores)

                important_index = (scores > mn).nonzero()
                sentence = np.array(sentence)

              
                import_token ={j:sentence[j] for j in important_index[0] if sentence[j].lower() not in self.stop_words  and sentence[j]  not in self.punctuations  }


                import_token={k: v for k, v in import_token.items() if len(v) > 2}
                candidates=defaultdict(list)
                for _, w in import_token.items():
                    cands = []
                    if w in self.counter:
                        cands = self.counter[w]

                    if len(cands):candidates[_] = cands

                if not len(candidates) :
                    continue

                indx_can = {k: len(v) for k, v in candidates.items()}
                main_ind = max(indx_can.items(), key=operator.itemgetter(1))[0]
                all_samples = [list(sentence.copy()) for j in range(len(candidates[main_ind]))]
                for _, word in enumerate(candidates[main_ind]):
                    all_samples[_][main_ind] = word
                    for j, v in candidates.items():
                        if j == main_ind: continue
                        all_samples[_][j] = random.sample(v, k=1)[0]

                all_samples=[' '.join(s) for s in all_samples][:1]
                tem ={'text': ' '.join(sentence), 'candidates':all_samples,'label':str(label)}
                data_to_save.append(tem)
            json.dump(data_to_save, open(self.opt.dataset_file['train_w_pl'], 'w'), indent=3)


    def model_train(self):

        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(self.model.parameters(), lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        if not os.path.exists(self.opt.dataset_file['train_w_pl']):
            train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
            self._generate_pseud_label(criterion, train_data_loader, val_data_loader)


        logger.info('loading clean and pseudo-labels data ')
        self.trainset = custom_dataset_pl(self.opt.dataset_file['train_w_pl'], self.tokenizer, self.opt.max_seq_len, self.opt.dataset, number_candidates=self.opt.n_candidates)
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=False)
        logger.info('ads and clean {}'.format(len(self.trainset)))
        logger.info('start training')


        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)


        self.model.load_state_dict(best_model_path)
        self.model.eval()
        pres, recall, f1_score, acc= self._evaluate_acc_f1(test_data_loader)

        logger.info(
            '>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format(pres, recall, f1_score, acc))
        with open('results_new.txt', 'a+') as f:
            f.write(' {} >> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}\n'.format( self.opt.dataset,pres, recall, f1_score, acc))
        f.close()




    def model_eval(self):

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        best_model_path ='state_dict/{}_{}_final_new.bm'.format(self.opt.dataset, self.opt.plm)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_pres,  test_recall,  test_f1_score,  test_acc= self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_precision: {:.4f}, test_recall: {:.4f}, test_f1: {:.4f}, test_acc: {:.4f}'.format( test_pres,  test_recall,  test_f1_score,  test_acc))

        pres, recall, f1_score, acc = self._evaluate_acc_f1(val_data_loader)
        logger.info('>> val_precision: {:.4f},  val_recall: {:.4f},  val_f1: {:.4f},  val_acc: {:.4f}'.format(pres, recall,f1_score, acc))




def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sst-2', type=str, help='agnews', choices=[ 'agnews', 'imdb', 'sst-2'])
    parser.add_argument('--method', default='train', type=str,  choices=['train', 'eval'])
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--logging_dir', default='result_log', type=str)
    parser.add_argument('--caching_dir', default='cache', type=str)
    parser.add_argument('--attack_method', default='textfooler', type=str)
    parser.add_argument('--save_path', default='state_dict/', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, )
    parser.add_argument('--adam_epsilon', default=2e-8, type=float, help='')
    parser.add_argument('--weight_decay', default=0, type=float )
    parser.add_argument('--negative_sampling', default=10, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--reg', type=float, default=0.00005)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--num_epoch_warming', default=5, type=int)
    parser.add_argument('--n_candidates', default=1, type=int, )
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_val', default=64, type=int)
    parser.add_argument('--batch-size-generate', default=8, type=int, help=' 2')
    parser.add_argument('--n_steps_interpret', default=70, type=int, help='50')
    parser.add_argument('--n_views', default=1, type=int, help='50')
    parser.add_argument('--log_step', default=500, type=int)
    parser.add_argument('--max_grad_norm', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.01, type=float)
    parser.add_argument('--temperature', default=0.05, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--plm', default='bert', type=str, choices=['bert', 'roberta', 'xlnet'])
    parser.add_argument('--pretrained_bert_name', default='bert', type=str)
    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--lebel_dim', default=2, type=int)
    parser.add_argument('--device', default='cuda' , type=str, help='e.g. cuda:0')
    parser.add_argument('--gpus', default='1' , type=str, help='e.g. cuda:0')
    parser.add_argument('--parrallel', default=0 , type=int, help=' parralel training ')
    parser.add_argument('--seed', default=65, type=int, help='set seed for reproducibility')
    parser.add_argument('--local_rank=1', default=-1, type=int)
    parser.add_argument('--inactivate_data_loss', action='store_true' )
    parser.add_argument('--m', default=0.3, type=float)
    parser.add_argument('--lamnd', default=0.1, type=float)

    opt = parser.parse_args()

    opt.pretrained_bert_name = '/workspace/plm/{}/'.format(opt.plm)


    if opt.dataset in ['imdb']:
        opt.lamnd =0.2
        opt.max_seq_len =256


    opt.seed= random.randint(20,300)

    if opt.seed is not None:

        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset_files = {
        'train': 'datasets/{0}/train.json'.format(opt.dataset),
        'train_w_pl': 'datasets/{}/train_w_pl.json'.format(opt.dataset),
        'test': 'datasets/{0}/test.json'.format(opt.dataset),
        'dev': 'datasets/{0}/dev.json'.format(opt.dataset)
    }

    input_colses =  ['input_ids', 'segments_ids', 'input_mask', 'label']
    input_colses_ads =  ['input_ids', 'segments_ids', 'input_mask','input_ids_c','segments_ids_c','input_mask_c', 'label']
    input_colses_val =  ['input_ids', 'segments_ids', 'input_mask', 'label']

    opt.dataset_file = dataset_files
    opt.inputs_cols = input_colses
    opt.inputs_cols_ads = input_colses_ads
    opt.inputs_cols_val = input_colses_val
    opt.initializer = torch.nn.init.xavier_uniform_
    opt.optimizer = AdamW
    opt.device = torch.device(opt.device if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    log_file = '{}-{}.log'.format(opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    logger.info('seed {}'.format(opt.seed))
    logger.info('dataset {} {}'.format(opt.dataset, opt.inactivate_data_loss))
    ins = Instructor(opt)
    if opt.method =='train':
        ins.model_train()
    elif opt.method =='eval':
        ins.model_eval()

   
if __name__ == '__main__':
    main()









