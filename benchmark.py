import random
import pickle
import torch
import numpy as np
from data_prepare import get_train_test

train_dict, test_dict = get_train_test()

Kshots= 5
epoch_num = 1
train_data = []
label_mapping = {0:0, 1:1, 2:2, 4:3}
is_sequence_training = True

for training_label in [0, 1, 2, 4]:
    candidates_ids = list(range(len(train_dict[training_label])))
    random.shuffle(candidates_ids)
    train_data.extend([(data_id, training_label) for data_id in candidates_ids[:Kshots]])
if not is_sequence_training:
    random.shuffle(train_data)

class FC(torch.nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.W1 = torch.nn.Parameter(torch.tensor(np.random.randn(4, 81*9)/ float(9.0), dtype=torch.float32))
        self.optimizer = torch.optim.Adam(self.parameters())
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.5)

    def train_step(self, feature, tgt=None):
        inp = torch.tensor(feature[:, :, 1:], dtype=torch.float32).reshape(-1, 9)
        kernel = torch.matmul(inp, inp.transpose(1, 0))
        attention_weights = torch.softmax(kernel, dim=1)
        attended_value = torch.matmul(attention_weights, inp)
        outs = torch.matmul(self.W1, attended_value.reshape(-1))
        probs = torch.softmax(outs, 0)

        if tgt is not None:
            tgt_oh = torch.zeros(4)
            tgt_oh[tgt] = 1.0
            loss = (- tgt_oh * torch.log(probs)).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return float(probs.argmax())

fc_model = FC()
for epoch in range(epoch_num):
    for data_id, data_label in train_data:
        input = train_dict[data_label][data_id]
        fc_model.train_step(input, tgt=label_mapping[data_label])

    if not is_sequence_training:
        random.shuffle(train_data)


prediction_correct_ct = 0
prediction_count = 0
print("testing dict size:" + str(len(test_dict[0])+len(test_dict[1])+len(test_dict[2])+len(test_dict[4])))
for testing_label in [0, 1, 2, 4]:
    for test_dt in test_dict[testing_label]:
        predict = fc_model.train_step(test_dt)
        correct = 1 if int(predict) == label_mapping[testing_label] else 0

        prediction_correct_ct += correct
        prediction_count += 1
        if prediction_count % 1000 == 0:
            print(prediction_count)

print("accuracy:" + str(prediction_correct_ct/float(prediction_count)))
print(prediction_count)
print("fin..")