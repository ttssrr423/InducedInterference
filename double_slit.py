import random
import numpy as np

period = 8
max_T = 60
emision_duration = 40
upper_threashold = 0.8
lower_threashold = -0.8
friend_scale = 0.1
stranger_scale = -0.01
foe_scale = -0.1
Kshot = 5

class Sources():
    def __init__(self, period, max_T):
        self.omega = 2*np.pi/float(period)
        rand_weights = 9.0*np.ones([2, 2, max_T])+np.random.random([2, 2, max_T]) # 2 slits * 2targets * duration_time
        self.route_weights = rand_weights / np.sum(rand_weights)

    def get_signals(self, length, is_inphase=None):
        phase_diff = np.pi if not is_inphase else 0.0
        diff_error = np.pi / 12.0

        signal_array = np.zeros([2,length, 2]) # 2 sources x time x 2 dim
        for tau in range(length):
            w1 = (0.5-random.random())*2.0 * diff_error + self.omega*tau
            w2 = (0.5-random.random()) * 2.0 + diff_error + phase_diff + self.omega * tau

            signal_array[0][tau][0] = np.sin(w1)
            signal_array[0][tau][1] = np.cos(w1)
            signal_array[1][tau][0] = np.sin(w2)
            signal_array[1][tau][1] = np.cos(w2)
        return signal_array

    def update_weights(self, delta_w):
        tmp_w = self.route_weights + delta_w
        renorm_sum = np.sum(tmp_w, axis=2, keepdims=True)
        self.route_weights = tmp_w / np.tile(renorm_sum, [1, 1, max_T])
        return

sources = Sources(period, max_T)

class Targets():
    def __init__(self, source_separation, target_separation):
        self.source_positions = [-source_separation//2, source_separation//2]
        self.target_positions = [-target_separation//2, target_separation//2]
        self.target_value = np.zeros([2])
        self.target_value[1] = 1.0

        route_availabilities = np.ones([2, 2, max_T, max_T])
        for si in range(2):
            for ti in range(2):
                dist = abs(self.target_positions[ti] - self.source_positions[si])
                for t in range(max_T):
                    for tau in range(max_T):
                        if t < tau+dist or tau >=emision_duration:
                            route_availabilities[si][ti][tau][t] = 0.0
        self.route_availabilities = route_availabilities
        return

    def learn(self, use_inphase=None, update_weights=None, sig_num=1000):
        signals = sources.get_signals(emision_duration, is_inphase=use_inphase)
        target_id = 1 if use_inphase else 0

        signals_reshaped = np.reshape(signals, [-1, 2])
        similarities_with_1 = np.matmul(signals_reshaped, np.expand_dims(self.target_value, axis=1))[:, 0]

        update_delta_weights = np.zeros([2, 2, max_T])
        J_on_ts = np.zeros([2, max_T])

        for t in range(max_T):
            arriving_routes = np.reshape(self.route_availabilities[:, :, :, t], [-1])
            valid_routes = list(np.argwhere(arriving_routes>0)[:, 0])
            if len(valid_routes) == 0:
                continue

            weights_updated_on_dur = np.zeros([2, 2, max_T])
            accumulating_targets = np.zeros([2, 2])
            accumulating_target_ct = np.zeros([2])
            accumulating_remainings = np.zeros([2, 2])

            for rid in valid_routes:
                st, tau = rid//max_T, rid % max_T
                source, tgt = st//2, st % 2
                travel_duration = t - tau
                # if tau >= emision_duration:
                #     continue

                similarity_arrived = similarities_with_1[source * emision_duration + tau]
                if similarity_arrived >= upper_threashold:
                    scale_factor = friend_scale
                    similarity_score = 1.0
                    J_on_ts[tgt][t] += similarity_score * sources.route_weights[source][tgt][travel_duration] * sig_num
                    accumulating_targets[tgt] += signals[source][tau]
                    accumulating_target_ct[tgt] += 1

                elif similarity_arrived < upper_threashold and similarity_arrived > lower_threashold:
                    scale_factor = stranger_scale
                    accumulating_remainings[tgt] += signals[source][tau] * sources.route_weights[source][tgt][travel_duration] * sig_num
                else:
                    scale_factor = foe_scale
                    accumulating_remainings[tgt] += signals[source][tau] * sources.route_weights[source][tgt][travel_duration] * sig_num

                if tgt == target_id:
                    weights_updated_on_dur[source][tgt][travel_duration] += sources.route_weights[source][tgt][travel_duration] * scale_factor # / float(emision_duration)


            for _tgt in range(2):
                attended_target_on_tgt = accumulating_targets[_tgt] / accumulating_target_ct[_tgt] if accumulating_target_ct[_tgt] > 0 else self.target_value
                remaining_energy = np.sum(accumulating_remainings[_tgt] * attended_target_on_tgt)
                J_on_ts[_tgt][t] += remaining_energy

            update_delta_weights += weights_updated_on_dur

        if update_weights:
            sources.update_weights(update_delta_weights)

        return J_on_ts, np.max(J_on_ts, axis=1)


learners = Targets(2, 4)

for i in range(Kshot):
    learners.learn(use_inphase=True, update_weights=True)
for i in range(Kshot):
    learners.learn(use_inphase=False, update_weights=True)

print("train finished")

total_ct = 0
correct_ct = 0
for i in range(200):
    tgt = random.randint(0, 1)
    is_inphase = (tgt == 1)
    J_on_ts, max_Js = learners.learn(use_inphase=is_inphase, update_weights=False)
    # pred = np.argmax(max_Js)
    pred = np.argmax(np.mean(J_on_ts, axis=1))

    if i % 20 == 0:
        print("evaluating :" + str(i) + "/200")
    if int(pred) == int(tgt):
        correct_ct += 1
    total_ct += 1

print(correct_ct / float(total_ct))
print("done")
