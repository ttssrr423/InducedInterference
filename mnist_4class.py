import numpy as np
import random
from data_prepare import get_train_test

train_dict, test_dict = get_train_test()

label_coords = {0:(-5, -5), 1:(5, -5), 2:(5, 5), 4:(-5, 5)}
target_id = {0:0, 1:1, 2:2, 4:3}

friend_scale = 1.0
stranger_scale = -0.5
foe_scale = -0.8
max_T = 25 # 18 is maximum travelling time  + 6 is tail duration
L = 9
emision_duration = 1
attention_threashold = 0.7
Kshots= 5
is_sequence_training = True
max_epoch=1

class Source(object):
    def __init__(self, i, j):
        self.x_coord, self.y_coord = float(i-(L-1)/2), float(j-(L-1)/2)
        self.id = i*L + j
        rand_weights = 9.0*np.ones([4, max_T])+np.random.random([4, max_T]) # destination num x travelling time duration
        self.route_weights = rand_weights
        for _c in range(4):
            self.route_weights[_c] = rand_weights[_c] / np.sum(rand_weights[_c])

    def update_weights(self, dw, tgt_id=None):
        if tgt_id is None:
            updating_label = np.argmax(np.sum(np.abs(dw), axis=1))
        else:
            updating_label = tgt_id
        tmp_w = dw + self.route_weights

        if tgt_id == -1:
            self.speed_weights = tmp_w / np.sum(tmp_w)
        else:
            self.route_weights[updating_label] = tmp_w[updating_label] / np.sum(tmp_w[updating_label])
        return

class Picture():
    srcs = []
    for i in range(L):
        for j in range(L):
            srcs.append(Source(i,j))

class Learner(object):
    def __init__(self, lab):
        self.x_coord, self.y_coord = label_coords[lab]
        tmp_value = np.array([0]+[1]*9)
        self.target_value = tmp_value / np.sqrt(np.sum(np.square(tmp_value)))
        self.target_id = target_id[lab]

        route_availabilities = np.ones([L*L, 1, max_T])
        for si in range(L*L):
            i, j = si//L, si%L
            x_coord, y_coord = float(i-(L-1)/2), float(j-(L-1)/2)
            dist = int(abs(x_coord-self.x_coord) + abs(y_coord-self.y_coord))
            for t in range(max_T):
                for tau in range(1):
                    if t < tau+dist or tau >=emision_duration:
                        route_availabilities[si][tau][t] = 0.0
        self.route_availabilities = route_availabilities
        return

    def learn_feature(self, feature, sig_num, update_weights=True):
        similarities_with_1 = np.matmul(np.reshape(feature, [L*L, 10]), np.expand_dims(self.target_value, axis=1))[:,0]

        lower_threashold = -attention_threashold
        upper_threashold = attention_threashold

        J_on_ts = np.zeros([max_T])
        update_delta_weights = np.zeros([L*L, 4, max_T])

        for t in range(max_T):
            arriving_routes = np.reshape(self.route_availabilities[:, 0, t], [-1])
            valid_routes = list(np.argwhere(arriving_routes>0)[:, 0])
            if len(valid_routes) == 0:
                continue

            weights_updated_on_dur = np.zeros([L*L, 4, max_T])
            accumulating_targets = np.zeros([10])
            accumulating_remainings = np.zeros([10])
            accumulating_target_ct = 0
            for si in valid_routes:
                tau = 0
                source = si
                i, j = si//L, si%L
                similarity_arrived = similarities_with_1[source * emision_duration + tau]
                travel_duration = t-tau

                if similarity_arrived >= upper_threashold:
                    scale_factor = friend_scale
                    similarity_score = 1.0
                    J_on_ts[t] += similarity_score * Picture.srcs[si].route_weights[self.target_id][travel_duration] * sig_num
                    accumulating_targets += feature[i][j]
                    accumulating_target_ct += 1
                elif similarity_arrived < upper_threashold and similarity_arrived > lower_threashold:
                    scale_factor = stranger_scale
                    accumulating_remainings += feature[i][j] * Picture.srcs[si].route_weights[self.target_id][travel_duration] * sig_num
                else:
                    scale_factor = foe_scale
                    accumulating_remainings += feature[i][j] * Picture.srcs[si].route_weights[self.target_id][travel_duration] * sig_num

                weights_updated_on_dur[source][self.target_id][travel_duration] += Picture.srcs[si].route_weights[self.target_id][travel_duration] * scale_factor

            attended_target_on_tgt = accumulating_targets / accumulating_target_ct if accumulating_target_ct > 0 else self.target_value
            remaining_energy = np.sum(accumulating_remainings * attended_target_on_tgt)
            J_on_ts[t] += remaining_energy

            update_delta_weights += weights_updated_on_dur

        if update_weights:
            for _id in range(L*L):
                Picture.srcs[_id].update_weights(update_delta_weights[_id], tgt_id=self.target_id)

        max_power = np.max(J_on_ts)
        power_peak = np.nanargmax(J_on_ts)
        energy_over_time = np.array([np.sum(J_on_ts[:t+1]) for t in range(max_T)])
        energy_peak = np.nanargmax(energy_over_time)

        return energy_over_time, max_power, power_peak, energy_peak

    def collect_white_signals(self, sig_num):
        # white signal input
        signal_over_time = np.ones([max_T, 4]) * 0.1
        for _id in range(max_T):
            i, j = _id // L, _id % L
            route_probabilities = Picture.srcs[_id].route_weights
            x_coord, y_coord = float(i-(L-1)/2), float(j-(L-1)/2)
            dist = int(abs(x_coord-self.x_coord) + abs(y_coord-self.y_coord))
            for t in range(dist, max_T):
                signal_over_time[t] += (sig_num * route_probabilities[:, t])

        time_distributions = signal_over_time / np.tile(np.sum(signal_over_time, axis=0, keepdims=True), [max_T, 1])
        weighted_peak = np.tile(np.expand_dims(np.arange(0, max_T), axis=1), [1, 4])
        weighted_peak_time = np.sum(time_distributions * weighted_peak, axis=0)
        entropy = (- time_distributions * np.log(time_distributions)).sum(axis=0)
        return weighted_peak_time, entropy

    def collect_arrival_similarity(self, feature, sig_num):
        population = np.zeros([max_T, L*L])
        for i in range(L):
            for j in range(L):
                _id = i*L+j
                x_coord, y_coord = float(i-(L-1)/2), float(j-(L-1)/2)
                dist = int(abs(x_coord-self.x_coord) + abs(y_coord-self.y_coord))
                minimum_time_arrival = dist
                for tau in range(emision_duration):
                    for t in range(minimum_time_arrival, max_T):
                        if tau > t:
                            continue
                        population[t][_id] = sig_num*Picture.srcs[_id].route_weights[self.target_id][t-tau]

        shifted_population = population+sig_num*0.0000001*np.ones_like(population)
        sampling_prob = population / np.tile(np.sum(shifted_population, axis=1, keepdims=True), [1, L*L])

        feat_reshape = np.reshape(feature, [L*L, 10])
        positionwise_similarity = np.matmul(feat_reshape, np.transpose(feat_reshape,[1, 0]))

        expected_similarity_over_t = np.zeros([max_T])
        for t in range(max_T):
            for i in range(L*L):
                for j in range(L*L):
                    weighted_similarity = sampling_prob[t][i] * sampling_prob[t][j] * positionwise_similarity[i][j]
                    expected_similarity_over_t[t] += weighted_similarity
        return expected_similarity_over_t


tgts = {}
train_data = []
for training_label in [0, 1, 2 ,4]:
    tgt = Learner(training_label)
    tgts[training_label] = tgt
    candidates_ids = list(range(len(train_dict[training_label])))
    random.shuffle(candidates_ids)
    train_data.extend([(data_id, training_label) for data_id in candidates_ids[:Kshots]])

if not is_sequence_training:
    random.shuffle(train_data)

for _ in range(max_epoch):
    for data_id, training_label in train_data:
        energy_over_time, max_power, power_peak, energy_peak = \
            tgts[training_label].learn_feature(train_dict[training_label][data_id], 1000)

test_cursor = {0:0, 1:0, 2:0, 4:0}
label_list = [0,1,2,4]
inference_entropy_matrix = np.zeros([4, 4])
test_sequence = {l:np.arange(len(test_dict[0])) for l in label_list}
for l in label_list:
    np.random.shuffle(test_sequence[l])
_si = 0

correct_count = 0
entropy_inferer_testing = np.zeros([4, 4])
metric_counter = np.zeros([4, 4])

while min([(test_cursor[l]-test_sequence[l].shape[0]) for l in label_list]) < 0:
    testing_label = label_list[_si % 4]
    if len(test_sequence[testing_label]) == test_cursor[testing_label]:
        continue
    feature_input = test_dict[testing_label][test_cursor[testing_label]]

    peak_obj = np.zeros([4])
    for inferer_label in label_list:
        energy_over_time, max_power, power_peak, energy_peak = \
            tgts[inferer_label].learn_feature(feature_input, 1000, update_weights=False)

        obj = max_power
        peak_obj[target_id[inferer_label]] = obj
    prediction = np.argmax(peak_obj)
    correctness = 1 if prediction == target_id[testing_label] else 0
    correct_count += correctness

    if _si % 100 == 0:
        if _si == 0:
            arrival_time_metric = np.zeros([4, 4])
            for infering_label in label_list:
                weighted_peak_time, entropy = tgts[infering_label].collect_white_signals(1000)
                arrival_time_metric[target_id[infering_label]] = weighted_peak_time
            print("white input metric for arrival time is:")
            print(arrival_time_metric)

        print(str(_si) + "  accuracy: " + str(correct_count/float(_si+1)))
    _si += 1
    test_cursor[testing_label]+=1

print(str(_si) + "  accuracy: " + str(correct_count/float(_si+1)))

expected_similarity_metric = np.zeros([4, 4])
for infering_label in label_list:
    for li in label_list:
        cross_similarity_tmp = []
        for _ in range(20):
            samp_id = random.randint(0, 500)
            feature_tmp = test_dict[li][samp_id]
            similarity_over_ts = tgts[infering_label].collect_arrival_similarity(feature_tmp,1000)
            cross_similarity_tmp.append(np.expand_dims(similarity_over_ts[-6:], axis=0))
        cross_similarity_entropy = np.mean(np.concatenate(cross_similarity_tmp, axis=0), axis=0)
        expected_similarity_metric[target_id[infering_label]][target_id[li]] = np.mean(cross_similarity_entropy[2:])
print("expected similarity among dipoles for different inferer-input combination:")
print(expected_similarity_metric)

print("done.")