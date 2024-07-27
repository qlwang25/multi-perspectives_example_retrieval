# coding=utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable


class CRF(nn.Module):
    # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    def __init__(self, num_tag, device):
        if num_tag <= 1:
            raise ValueError("Invalid value of num_tag: %d" % num_tag)
        super(CRF, self).__init__()
        self.tagset_size = num_tag
        self.START_ID = num_tag
        self.STOP_ID = num_tag + 1
        self.PAD_ID = num_tag - 1
        print("CRF PAD TOKEN ID is {}".format(self.PAD_ID))
        self.device = device

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size + 2, self.tagset_size + 2))

        # These two statements enforce the constraint that we never transfer to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_ID, :] = -10000
        self.transitions.data[:, self.STOP_ID] = -10000

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        # init_alphas[0][self.START_ID] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas

        # Iterate through the sentence
        for i, feat in enumerate(feats):
            if i == 0:
                forward_var = self.transitions[:self.tagset_size, self.START_ID] + feat
                forward_var = forward_var.view(1, -1)
            else:
                forward_var = forward_var.repeat(self.tagset_size, 1)

                emit_score = feat.view(-1, 1).repeat(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[:self.tagset_size, :self.tagset_size]
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                next_var = forward_var + trans_score + emit_score
                forward_var = self._log_sum_exp(next_var).view(1, -1)

        terminal_var = forward_var + self.transitions[self.STOP_ID, :self.tagset_size]
        alpha = self._log_sum_exp(terminal_var.view(1, -1))
        return alpha

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.START_ID], dtype=torch.long).to(self.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.STOP_ID, tags[-1]]
        return score

    def _log_sum_exp(self, matrix):
        # Compute log sum exp in a numerically stable way for the forward algorithm
        max_score, _ = torch.max(matrix, dim=1)
        max_score_broadcast = max_score.view(-1, 1).repeat(1, self.tagset_size)
        log_sum = max_score + torch.log(torch.sum(torch.exp(matrix - max_score_broadcast), dim=1))
        return log_sum

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        # init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # init_vvars[0][self.START_ID] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        # forward_var = init_vvars
        for i, feat in enumerate(feats):
            if i == 0:
                next_var = self.transitions[:self.tagset_size, self.START_ID] + feat
                forward_var = next_var.view(1, -1)
                backpointers.append([self.START_ID] * self.tagset_size)
            else:
                next_var = forward_var.repeat((self.tagset_size, 1)) + self.transitions[:self.tagset_size, :self.tagset_size]
                viterbivar, best_tag_id = torch.max(next_var, dim=1)         

                # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed
                forward_var = (viterbivar + feat).view(1, -1)
                backpointers.append(best_tag_id.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var.squeeze(0) + self.transitions[self.STOP_ID, :self.tagset_size]
        best_tag_id = torch.argmax(terminal_var, dim=0).item()
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.START_ID  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, feats_mask, tags):
        loss = torch.zeros(1).to(self.device)
        num_tag = feats.size(2)
        num_chars = torch.sum(feats_mask.detach()).float()
        if num_chars.item() == 0.0:
            raise ValueError
        for ix, (feat, tag) in enumerate(zip(feats, tags)):
            feat = feat[feats_mask[ix]==1]
            tag = tag[feats_mask[ix]==1]

            total_score = self._forward_alg(feat)
            real_score = self._score_sentence(feat, tag)

            cost = total_score - real_score
            loss += cost

        ave_loss = loss / num_chars
        return ave_loss

    def forward(self, feats, feats_mask):
        # dont confuse this with _forward_alg above.
        batch_best_path, batch_best_score = [], []
        batch_size, max_len, num_tag = feats.size()
        for ix, feat in enumerate(feats):
            feat = feat[feats_mask[ix]==1]
            
            # Find the best path, given the features.
            score, tag_seq = self._viterbi_decode(feat)
            tag_seq = torch.Tensor(tag_seq).long().to(self.device)
            
            # padding
            vec = torch.cat([tag_seq, torch.full((max_len-tag_seq.size(0), ), self.PAD_ID).long().to(self.device)], dim=0)
            batch_best_path.append(vec)
            batch_best_score.append(score)

        batch_best_path = torch.stack(batch_best_path, dim=0)
        batch_best_score = torch.stack(batch_best_score, dim=0)
        return batch_best_path, batch_best_score
