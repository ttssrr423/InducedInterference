import numpy as np
from scipy.optimize import fsolve

def get_dist(x1, y1, x2, y2, vertical_dist=1):
    return abs(x1-x2) + abs(y1-y2) + vertical_dist

class Source():
    def __init__(self, T):
        self.T = T
        return

class SeparateSource(Source):
    def __init__(self, T, patch_size, coords):
        super().__init__(T)
        self.data = np.zeros([T, 1, len(coords), patch_size*patch_size])
        self.width = len(coords)
        self.height = 1
        self.coords = [coords]

    def ob_from_vid(self, pic_stream):
        assert pic_stream.shape[0] == self.T and pic_stream.shape[1] == self.height and pic_stream.shape[2] == self.width
        for t in range(self.T):
            for i in range(self.height):
                for j in range(self.width):
                    self.data[t][i][j] = pic_stream[t][i][j]
        return

    def ob_pic_frames(self, pic, start, end):
        assert pic.shape[0] == self.height and pic.shape[1] == self.width
        for t in range(start, end):
            for i in range(len(self.coords)):
                for j in range(len(self.coords[0])):
                    _x, _y = self.coords[i][j]
                    self.data[t][i][j] = pic[i][j]
        return

class SourceArray(Source):
    def __init__(self, T, patch_size, field_size):
        super().__init__(T)
        assert patch_size % 2 == 1
        self.r = r = (patch_size-1) // 2
        self.coords = []
        self.field_size = field_size
        self.height = field_size // patch_size
        self.width = field_size // patch_size
        for i in range(field_size // patch_size):
            self.coords.append([])
            for j in range(field_size // patch_size):
                self.coords[i].append((r+i*patch_size, r+j*patch_size))

        self.data = np.zeros([T, len(self.coords), len(self.coords[0]), patch_size*patch_size])
        return

    def ob_from_vid(self, pic_stream):
        assert pic_stream.shape[0] == self.T and pic_stream.shape[1] == self.field_size and pic_stream.shape[2] == self.field_size
        for t in range(self.T):
            for i in range(len(self.coords)):
                for j in range(len(self.coords[0])):
                    _x, _y = self.coords[i][j]
                    self.data[t][i][j] = np.reshape(pic_stream[t][_x-self.r:_x+1+self.r, _y-self.r:_y+1+self.r], [-1])
        return

    def ob_pic_frames(self, pic, start, end):
        assert pic.shape[0] == self.field_size and pic.shape[1] == self.field_size
        for t in range(start, end):
            for i in range(len(self.coords)):
                for j in range(len(self.coords[0])):
                    _x, _y = self.coords[i][j]
                    self.data[t][i][j] = np.reshape(pic[_x-self.r:_x+1+self.r, _y-self.r:_y+1+self.r], [-1])
        return

class Inducer():
    def __init__(self, feature_size, low_feat_source, high_feat_source, local_mf_coords):
        assert feature_size == low_feat_source.data.shape[-1] and feature_size == high_feat_source.data.shape[-1]
        self.l_height, self.l_width = low_feat_source.height, low_feat_source.width
        self.h_height, self.h_width = 1, len(local_mf_coords)
        self.l_source = low_feat_source
        self.h_source = high_feat_source
        self.l_paths = Paths(low_feat_source, self, True)
        self.h_paths = Paths(high_feat_source, self, False)
        self.prev_n_bars = None
        self.n_bars = {}
        self.aggregates = [[] for _ in range(len(local_mf_coords))]
        self.mf_coords = [local_mf_coords]
        self.full_emit_rate = 100.0
        self.base_emit_rate = 100.0
        self.max_neighbour = 1.5
        self.critic_roots = {}

        for i in range(len(self.l_source.coords)):
            for j in range(len(self.l_source.coords[0])):
                l_x, l_y = self.l_source.coords[i][j]
                min_dist = 99999
                min_k = -1
                for k in range(len(self.mf_coords[0])):
                    h_x, h_y = self.mf_coords[0][k]
                    dist = get_dist(l_x, l_y, h_x, h_y, vertical_dist=0)
                    if dist < min_dist:
                        min_k = k
                        min_dist = dist
                self.aggregates[min_k].append((i, j))

        for k in range(len(self.aggregates)):
            x1, y1 = self.mf_coords[0][k]
            for source_k in range(self.h_source.width):
                x2, y2 = self.h_source.coords[0][source_k]
                dist_high = get_dist(x1, y1, x2, y2)
                self.h_paths.min_travel_duration[0][source_k][0][k] = dist_high

        for i in range(len(self.l_source.coords)):
            for j in range(len(self.l_source.coords[0])):
                x1, y1 = self.l_source.coords[i][j]
                for i2 in range(len(self.l_source.coords)):
                    for j2 in range(len(self.l_source.coords[0])):
                        x2, y2 = self.l_source.coords[i2][j2]
                        dist = get_dist(x1, y1, x2, y2)
                        self.l_paths.min_travel_duration[i][j][i2][j2] = dist
        return

    def collect_from_low(self, t):
        source = self.l_source
        local_collections = {}

        for ir in range(len(source.coords)):
            for jr in range(len(source.coords[0])):
                collected_signals = []
                collected_signal_intensities = []
                collected_patches = []
                for i in range(len(source.coords)):
                    for j in range(len(source.coords[0])):
                        if t < self.l_paths.min_travel_duration[i,j,ir,jr]:
                            continue
                        latest_emission = int(t-self.l_paths.min_travel_duration[i,j,ir,jr])
                        possible_arrival_signal = (source.data[:latest_emission+1])[:, i, j, :]
                        travel_durations = t - np.arange(latest_emission+1)
                        arrived_signal_probs = (self.l_paths.durations[travel_durations])[:, i, j, ir, jr]
                        # positive_prob = (possible_arrival_signal + 1) / 2.0
                        # negative_prob = 1.0 - positive_prob
                        # emit rate is proportaional to mean pixel darkness
                        emit_rate = ((np.mean(possible_arrival_signal)/2.0+0.5) * (self.full_emit_rate-self.base_emit_rate)) + self.base_emit_rate
                        arrived_intensity = emit_rate * arrived_signal_probs
                        collected_signal_intensities.append(arrived_intensity)
                        collected_signals.append(possible_arrival_signal)
                        emit_tm_expanded = np.expand_dims(np.arange(latest_emission+1), axis=1)
                        emit_patch_id = np.concatenate([emit_tm_expanded, i*np.ones_like(emit_tm_expanded), j*np.ones_like(emit_tm_expanded)], axis=1)
                        collected_patches.append(emit_patch_id)

                if len(collected_signals) > 1:
                    collected_signals_cat = np.concatenate(collected_signals, axis=0)
                    collected_signal_intensities_cat = np.concatenate(collected_signal_intensities, axis=0)
                    collected_patches_cat = np.concatenate(collected_patches, axis=0)
                else:
                    collected_signals_cat = collected_signals[0]
                    collected_signal_intensities_cat = collected_signal_intensities[0]
                    collected_patches_cat = collected_patches[0]

                if ir not in local_collections:
                    local_collections[ir] = {}

                local_collections[ir][jr] = (collected_patches_cat, collected_signals_cat, collected_signal_intensities_cat)
                # local_collections.append((ir, jr, collected_patches_cat, collected_signals_cat, collected_signal_intensities_cat))

        return local_collections

    def collect_from_high(self, t):
        local_collections = []
        source = self.h_source
        for kr in range(len(self.mf_coords[0])):
            collected_signals = []
            collected_signal_intensities = []
            collected_patches = []
            for k in range(len(source.coords[0])):
                min_travel_duration = get_dist(self.mf_coords[0][kr][0], self.mf_coords[0][kr][1],
                                               source.coords[0][k][0], source.coords[0][k][1])
                if t < min_travel_duration:
                    continue
                latest_emission = int(t-self.h_paths.min_travel_duration[0, k, 0, kr])
                possible_arrival_signal = (source.data[:latest_emission+1])[:, 0, k, :]
                travel_durations = t - np.arange(latest_emission+1)
                arrived_signal_probs = (self.h_paths.durations[travel_durations])[:, 0, k, 0, kr]
                emit_rate = ((np.mean(possible_arrival_signal)/2.0+0.5) * (self.full_emit_rate-self.base_emit_rate)) + self.base_emit_rate
                arrived_intensity = emit_rate * arrived_signal_probs
                collected_signal_intensities.append(arrived_intensity)
                collected_signals.append(possible_arrival_signal)
                emit_tm_expanded = np.expand_dims(np.arange(latest_emission+1), axis=1)
                emit_patch_id = np.concatenate([emit_tm_expanded, 0*np.ones_like(emit_tm_expanded), k*np.ones_like(emit_tm_expanded)], axis=1)
                collected_patches.append(emit_patch_id)

            if len(collected_signals) == 0:
                return []
            elif len(collected_signals) > 1:
                collected_signals_cat = np.concatenate(collected_signals, axis=0)
                collected_signal_intensities_cat = np.concatenate(collected_signal_intensities, axis=0)
                collected_patches_cat = np.concatenate(collected_patches, axis=0)
            else:
                collected_signals_cat = collected_signals[0]
                collected_signal_intensities_cat = collected_signal_intensities[0]
                collected_patches_cat = collected_patches[0]

            local_collections.append((0, kr, collected_patches_cat, collected_signals_cat, collected_signal_intensities_cat))

        return local_collections

    def calc_low_local_mf(self, li, lj, low_collected_patches, low_collected_signals, low_collected_intensities, external_field_density):
        positive_prob = (low_collected_signals + 1) / 2.0
        negative_prob = 1 - positive_prob

        positive_neuron_number = positive_prob*np.expand_dims(low_collected_intensities, axis=1)
        negative_neuron_number = negative_prob*np.expand_dims(low_collected_intensities, axis=1)
        positive_ratio = np.sum(positive_neuron_number) / (np.sum(negative_neuron_number) + np.sum(positive_neuron_number))
        negative_ratio = 1.0 - positive_ratio
        neuron_pair_energy_expectation = positive_ratio * (positive_ratio - negative_ratio) + negative_ratio*(negative_ratio - positive_ratio)

        # neuron_pair_energy_expectation*9 == internal_energy_expectation # should be True

        ratio_combinations = [positive_ratio*positive_ratio, negative_ratio*negative_ratio, 2*positive_ratio*negative_ratio]
        reinforce_scale = 1.2 # low patch self-interaction, preferes more on positive-positive interaction
        retain_scale = 1.0
        degenerate_scale = 0.8
        connection_ratio_evolved = [ratio_combinations[0]*reinforce_scale, ratio_combinations[1]*retain_scale, ratio_combinations[2]*degenerate_scale]
        connection_ratio_evolved = connection_ratio_evolved/(sum(connection_ratio_evolved))
        neuron_pair_energy_attented_expectation = connection_ratio_evolved[0] + connection_ratio_evolved[1] - connection_ratio_evolved[2]

        n_bar = neuron_pair_energy_attented_expectation * self.max_neighbour # assuming maximum number of neighbours achieved when all signals are in-phase
        if li not in self.n_bars: self.n_bars[li] = {}
        self.n_bars[li][lj] = n_bar
        if self.prev_n_bars is not None and li in self.prev_n_bars and lj in self.prev_n_bars:
            momentum = 0.2 # every low patch has its own self-attn connection n_bar. This prevents critical state to appear at intervals.
            n_bar = n_bar * momentum + (1.0-momentum)*self.prev_n_bars[li][lj]

        def mf_eq(m):
            return np.tanh(n_bar * m + external_field_density) - m

        mf_roots = fsolve(mf_eq, (-2.0, 0.0, 2.0)) # roots evaluated from 3 different initial points
        return mf_roots, n_bar, neuron_pair_energy_expectation

    def get_critic_field_solutions(self, n_bar, low_collected_intensities):
        n_bar_discrete = (int(n_bar * 100) / 100.0)

        def find_critic(b_ext, n_bar_discrete):
            def solve_mf(m):
                return np.tanh(n_bar_discrete * m + b_ext) - m
            return solve_mf

        if n_bar_discrete not in self.critic_roots:
            min_epsilon_pos = 1.0
            argmin_epsilon_b_ext_pos = 0.0
            min_epsilon_neg = 1.0
            argmin_epsilon_b_ext_neg = 0.0
            for b_ext_i in range(200):
                try_b_ext = -1.0 + ((b_ext_i+1)/100.0)
                critic_fn = find_critic(try_b_ext, n_bar_discrete)
                critic_roots = fsolve(critic_fn, (-2.0, 0.0, 2.0))
                if abs(critic_roots[0] - critic_roots[1]) > 0.001 and abs(critic_roots[1] - critic_roots[2]) < 0.1 and try_b_ext <= 0.0:
                    epsilon = abs(critic_roots[1] - critic_roots[2])
                    if epsilon < min_epsilon_neg:
                        min_epsilon_neg = epsilon
                        argmin_epsilon_b_ext_neg = try_b_ext
                if abs(critic_roots[1] - critic_roots[2]) > 0.001 and abs(critic_roots[0] - critic_roots[1]) < 0.1 and try_b_ext > 0.0:
                    epsilon = abs(critic_roots[0] - critic_roots[1])
                    if epsilon < min_epsilon_pos:
                        min_epsilon_pos = epsilon
                        argmin_epsilon_b_ext_pos = try_b_ext

            critic_root_weight = np.sum(low_collected_intensities) # give low patch with more collected signals a larger weight in critic target voting process?
            self.critic_roots[n_bar_discrete] = (argmin_epsilon_b_ext_neg, argmin_epsilon_b_ext_pos, critic_root_weight, n_bar_discrete)
        local_target_b_ext = self.critic_roots[n_bar_discrete]
        return local_target_b_ext

    def calculate_low_path_updates(self, mf_roots, n_bar, t, li, lj, low_collected_patches, low_collected_signals, low_collected_intensities):
        """
        ### Unsupervised learning condition is whether constructive interference locally exceeds a threashold.
        ### When threashold exceeds, n_bar > 1.0, and 3 mean field roots can be solved.
        ### Choose the root which is closer to most of the arrived signals.
        ### Update path probabilities based on whether the arrival of a signal could enlarge the gap between positive and negative roots.
        ### We let the signals' arrivel could increase the constructive interference strength.
        ### Due to the process of reinforcing constructive interference, the time accumulated signal intensity
        ### is expected(also observed) to increases in inductable low patch, and in contrast the accumulated signal intensity
        ### decreases in non-inductable low patch.
        """
        if abs(mf_roots[0] - mf_roots[1]) > 0.001 or abs(mf_roots[0] - mf_roots[2])>0.001 or abs(mf_roots[1] - mf_roots[2])>0.001:
            # print("phase change could occure")
            critical_patch_ct_plus = 1
            learning_rate = 0.2
            # find B_ext for critical state, let the B_ext* critical solutiions vote for a suitable target to update target interference.
            local_target_b_ext = self.get_critic_field_solutions(n_bar, low_collected_intensities)
        else:
            critical_patch_ct_plus = 0
            learning_rate = 0.0
            local_target_b_ext = None

        low_collected_signals_repeated = np.tile(np.expand_dims(low_collected_signals, axis=0), [3, 1, 1])
        roots_expanded = np.tile(np.expand_dims(np.expand_dims(mf_roots, axis=-1), axis=-1), [1, low_collected_signals_repeated.shape[1], low_collected_signals_repeated.shape[2]])
        matchness_low_sig_and_roots = np.mean(np.abs(low_collected_signals_repeated - roots_expanded), axis=-1)
        chosen_root_id = np.argmin(np.mean(matchness_low_sig_and_roots, axis=1))
        collapsed_mf = mf_roots[chosen_root_id]
        if collapsed_mf > 0.0 and critical_patch_ct_plus>0:
            path_prob_scale = np.where(np.mean(low_collected_signals, axis=-1) >= collapsed_mf,
                                       learning_rate*np.ones_like(low_collected_intensities),
                                       -learning_rate*np.ones_like(low_collected_intensities))
        elif collapsed_mf < 0.0 and critical_patch_ct_plus>0:
            path_prob_scale = np.where(np.mean(low_collected_signals, axis=-1) <= collapsed_mf,
                                       learning_rate*np.ones_like(low_collected_intensities),
                                       -learning_rate*np.ones_like(low_collected_intensities))
        else:
            path_prob_scale = np.zeros_like(low_collected_intensities)

        interacting_population_num = low_collected_patches.shape[0]
        cource_tgt_indicies = np.concatenate([low_collected_patches,
                                              li*np.ones([interacting_population_num, 1], dtype=np.int32),
                                              lj*np.ones([interacting_population_num, 1], dtype=np.int32)], axis=1)

        update_delta = np.zeros_like(self.l_paths.durations)
        for bi in range(interacting_population_num):
            update_delta[t-cource_tgt_indicies[bi,0], cource_tgt_indicies[bi,1], cource_tgt_indicies[bi,2], cource_tgt_indicies[bi,3], cource_tgt_indicies[bi,4]] = path_prob_scale[bi]

        self.l_paths.register_update(update_delta)

        return critical_patch_ct_plus, local_target_b_ext, collapsed_mf

    def balance_b_ext_for_critical(self, aggregated_low_patch_mean_fields):
        for local_low_critic_bexts, local_low_intensity, local_collapsed_mf, local_low_nbar in aggregated_low_patch_mean_fields:
            local_critic_bext_pos, local_critic_bext_neg = local_low_critic_bexts[1], local_low_critic_bexts[0]
            """
            ### Negative collapsed mf is endangered if b_ext shifts y=x line towards right, corresponding to negative local_critic_bext.
            ### Any sensitive change in n_bar of the voted low patch, could result in great change in the collapsed mean field.
            ### If n_bar is greater than critical condition, collapsed mf would remain roughly same;
            ### however if n_bar is not large enough, tanh would be slightly flattened and the only valid root would phase change to the opposite sign.
            ### Super-critical condition allows a small gap to exist before phase change takes place as n_bar gradually decrease.
            ### The balancer is trying to get as much of the low patch signals into a close-critical state as possible by choosing a suitable b_ext.
            ### The method of adjusting b_ext is to update h_path probabilities and use interference from random targets to construct a b_ext.
            ### A problem remains unsolved: should we adjust b_ext to maintain critical for low patches only based on signal intensity?
            ### Is there any other aspects we should consider to decide the low patches' voting weights? 
            ### i.e. ignoring those patches are already well trained if they are too much in a  under-critical state. 
            """
            voting_bext = local_critic_bext_neg if local_collapsed_mf < 0 else local_critic_bext_pos




        # critic_bext_tgt_proporsals[0] <- (argmin_epsilon_b_ext_neg, argmin_epsilon_b_ext_pos, critic_root_weight, n_bar_discrete)
        pos_part = np.array([x[1] for x in critic_bext_tgt_proporsals]) # x[0] == -x[1] should be True
        critic_solution_histogram, linespace_intervals = np.histogram(pos_part, bins=20)
        weight_stat = np.zeros_like(critic_solution_histogram, dtype=np.float)
        for item in critic_bext_tgt_proporsals:
            root_weight = item[2]
            root_value = item[1]
            for _i in range(weight_stat.shape[0]):
                if root_value >= linespace_intervals[_i] and root_value < linespace_intervals[_i+1]:
                    weight_stat[_i] += root_weight

        most_critic_id = np.argmax(weight_stat) # choose the most weighted (most low patch signals responding to) critic B_ext
        critic_lb = linespace_intervals[most_critic_id]-0.5*(linespace_intervals[most_critic_id+1]-linespace_intervals[most_critic_id])
        critic_hb = linespace_intervals[most_critic_id]+0.5*(linespace_intervals[most_critic_id+1]-linespace_intervals[most_critic_id])
        tgt_field_pos = np.random.random([high_collected_signals.shape[1]])*(critic_hb-critic_lb) + critic_lb
        tgt_field_neg = -tgt_field_pos
        two_roots = np.concatenate([tgt_field_pos[None], tgt_field_neg[None]], axis=0)
        amendment_candidates = two_roots - direct_mean_external_field_density
        matchness_to_amendment = np.abs(np.tile(np.expand_dims(high_collected_signals, axis=1), [1, 2, 1]) - amendment_candidates[None])
        amendment_inefficientness = np.mean(np.mean(matchness_to_amendment, axis=-1), axis=0)
        amendment_chosen_id = np.argmin(amendment_inefficientness)
        chosen_amendment = amendment_candidates[amendment_chosen_id]

        update_delta_h_path = np.zeros_like(self.h_paths.durations)
        for hbi in range(high_collected_patches.shape[0]):
            signal_matchness_with_amendment = np.mean(matchness_to_amendment[hbi, amendment_chosen_id]) # matchness:[0.0, 2.0], small is good match
            lr_h_based_on_amendment_matchness = (1.0 - signal_matchness_with_amendment) * 0.5
            update_delta_h_path[t-high_collected_patches[hbi, 0], high_collected_patches[hbi,1], high_collected_patches[hbi, 2], high_i, high_j] = lr_h_based_on_amendment_matchness
        self.h_paths.register_update(update_delta_h_path)
        return

    def local_mean_field_induction(self, local_high_collections, local_low_collections, t, maintain_critic=True):
        induction_local_energies = []

        if len(local_high_collections) == 0:
            critic_patch_evaluated = {}
            critical_patch_ct = 0
            critic_local_field_intensity_stat = [0.0, 0.0]
            for li in range(self.l_source.height):
                for lj in range(self.l_source.width):
                    low_collected_patches, low_collected_signals, low_collected_intensities = local_low_collections[li][lj]
                    mf_roots, n_bar, neuron_pair_energy_expectation = self.calc_low_local_mf(li, lj, low_collected_patches, low_collected_signals, low_collected_intensities, 0.0)
                    induction_local_energies.append(neuron_pair_energy_expectation)
                    critical_patch_ct_plus, local_target_b_ext, collapsed_mf = self.calculate_low_path_updates(mf_roots, n_bar, t, li, lj, low_collected_patches, low_collected_signals, low_collected_intensities)
                    if local_target_b_ext is not None:
                        if li not in critic_patch_evaluated:
                            critic_patch_evaluated[li] = {}
                        critical_intensity = np.sum(low_collected_intensities)
                        critic_patch_evaluated[li][lj] = (local_target_b_ext, critical_intensity, collapsed_mf)
                        critical_patch_ct += critical_patch_ct_plus
                        critic_local_field_intensity_stat[0] += critical_intensity
                    else:
                        critic_local_field_intensity_stat[1] += np.sum(low_collected_intensities)

            if critical_patch_ct > 0 and maintain_critic:
                for k in range(self.h_width):
                    high_patch_observed_bexts = []
                    for li, lj in self.aggregates[k]:
                        if li in critic_patch_evaluated and lj in critic_patch_evaluated[li]:
                            high_patch_observed_bexts.append((*critic_patch_evaluated[li][lj], self.n_bars[li][lj]))
                    if len(high_patch_observed_bexts) > 0:
                        self.balance_b_ext_for_critical(high_patch_observed_bexts)

            local_self_attention_energy_averaged = sum(induction_local_energies) / float(len(induction_local_energies))
            return critical_patch_ct, local_self_attention_energy_averaged, critic_local_field_intensity_stat


        high_patch_energies = []
        for high_i, high_j, high_collected_patches, high_collected_signals, high_collected_intensities in local_high_collections:
            critical_patch_ct = 0

            composite_low_patches = self.aggregates[high_j]

            h_positive_prob = (high_collected_signals + 1) / 2.0
            h_negative_prob = 1 - high_collected_signals
            h_positive_neuron_number = h_positive_prob*np.expand_dims(high_collected_intensities, axis=1)
            h_negative_neuron_number = h_negative_prob*np.expand_dims(high_collected_intensities, axis=1)
            h_positive_ratio = np.sum(h_positive_neuron_number) / (np.sum(h_negative_neuron_number) + np.sum(h_positive_neuron_number))
            h_negative_ratio = 1.0 - h_positive_ratio
            h_neuron_pair_energy_expectation = h_positive_ratio * (h_positive_ratio - h_negative_ratio) + h_negative_ratio*(h_negative_ratio - h_positive_ratio)
            high_patch_energies.append(h_neuron_pair_energy_expectation)

            # external_field_strengths = np.expand_dims(high_collected_intensities, axis=-1) * high_collected_signals
            # # direct_mean_external_field = np.mean(external_field_strengths, axis=0)
            # high_signal_dot_energies = np.matmul(high_collected_signals, np.transpose(high_collected_signals, [1, 0]))
            # h_signal_chosen_prob = high_collected_intensities[None]/ np.sum(high_collected_intensities)
            # pair_h_signal_prob = np.matmul(np.transpose(h_signal_chosen_prob, [1, 0]), h_signal_chosen_prob)
            # internal_energy_high = np.sum(high_signal_dot_energies*pair_h_signal_prob)
            # high_patch_energies.append(internal_energy_high / high_collected_signals.shape[1])
            # direct_mean_external_field_density = np.sum(np.mean(external_field_strengths, axis=1), axis=0) / np.sum(high_collected_intensities)
            direct_mean_external_field_density = h_neuron_pair_energy_expectation

            critic_bext_tgt_proporsals = []

            for low_patch in composite_low_patches:
                li, lj = low_patch
                low_collected_patches, low_collected_signals, low_collected_intensities = local_low_collections[li][lj]
                # internal_dots = np.matmul(low_collected_signals, np.transpose(low_collected_signals, [1, 0]))
                # signal_chosen_prob = low_collected_intensities[None] / np.sum(low_collected_intensities)
                # pair_signal_prob = np.matmul(np.transpose(signal_chosen_prob, [1, 0]), signal_chosen_prob)
                # energy expectation between any 2 vector signals
                # internal_energy_expectation = np.sum(internal_dots * pair_signal_prob)
                mf_roots, n_bar, neuron_pair_energy_expectation = self.calc_low_local_mf(li, lj, low_collected_patches, low_collected_signals, low_collected_intensities, direct_mean_external_field_density)
                induction_local_energies.append(neuron_pair_energy_expectation)

                critical_patch_ct_plus, local_target_b_ext, collapsed_mf = self.calculate_low_path_updates(mf_roots, n_bar, t, li, lj, low_collected_patches, low_collected_signals, low_collected_intensities)
                if local_target_b_ext is not None:
                    critic_bext_tgt_proporsals.append(local_target_b_ext)
                    critical_patch_ct += critical_patch_ct_plus

            # ------- supervised training -----------
            if len(critic_bext_tgt_proporsals) > 0:
                # there exist critic states in some of the composite low patches.
                pos_part = np.array([x[1] for x in critic_bext_tgt_proporsals]) # x[0] == -x[1] should be True
                critic_solution_histogram, linespace_intervals = np.histogram(pos_part, bins=20)
                weight_stat = np.zeros_like(critic_solution_histogram, dtype=np.float)
                for item in critic_bext_tgt_proporsals:
                    root_weight = item[2]
                    root_value = item[1]
                    for _i in range(weight_stat.shape[0]):
                        if root_value >= linespace_intervals[_i] and root_value < linespace_intervals[_i+1]:
                            weight_stat[_i] += root_weight

                most_critic_id = np.argmax(weight_stat) # choose the most weighted (most low patch signals responding to) critic B_ext
                critic_lb = linespace_intervals[most_critic_id]-0.5*(linespace_intervals[most_critic_id+1]-linespace_intervals[most_critic_id])
                critic_hb = linespace_intervals[most_critic_id]+0.5*(linespace_intervals[most_critic_id+1]-linespace_intervals[most_critic_id])
                tgt_field_pos = np.random.random([high_collected_signals.shape[1]])*(critic_hb-critic_lb) + critic_lb
                tgt_field_neg = -tgt_field_pos
                two_roots = np.concatenate([tgt_field_pos[None], tgt_field_neg[None]], axis=0)
                amendment_candidates = two_roots - direct_mean_external_field_density
                matchness_to_amendment = np.abs(np.tile(np.expand_dims(high_collected_signals, axis=1), [1, 2, 1]) - amendment_candidates[None])
                amendment_inefficientness = np.mean(np.mean(matchness_to_amendment, axis=-1), axis=0)
                amendment_chosen_id = np.argmin(amendment_inefficientness)
                chosen_amendment = amendment_candidates[amendment_chosen_id]

                update_delta_h_path = np.zeros_like(self.h_paths.durations)
                for hbi in range(high_collected_patches.shape[0]):
                    signal_matchness_with_amendment = np.mean(matchness_to_amendment[hbi, amendment_chosen_id]) # matchness:[0.0, 2.0], small is good match
                    lr_h_based_on_amendment_matchness = (1.0 - signal_matchness_with_amendment) * 0.5
                    update_delta_h_path[t-high_collected_patches[hbi, 0], high_collected_patches[hbi,1], high_collected_patches[hbi, 2], high_i, high_j] = lr_h_based_on_amendment_matchness
                self.h_paths.register_update(update_delta_h_path)
            else:
                update_delta_h_path = np.zeros_like(self.h_paths.durations)
                self.h_paths.register_update(update_delta_h_path)


        local_self_attention_energy_averaged = sum(induction_local_energies) / float(len(induction_local_energies))
        high_patch_energy_averaged = sum(high_patch_energies) / float(len(high_patch_energies))
        return critical_patch_ct, local_self_attention_energy_averaged, high_patch_energy_averaged

    def update_paths(self):
        self.l_paths.apply_updates()
        self.h_paths.apply_updates()
        self.prev_n_bars = self.n_bars
        return

class Paths():
    def __init__(self, source, inducer, is_low_abstraction):
        self.source = source
        self.inducer = inducer
        self.inducer_connnect_type = "low_end" if is_low_abstraction else "high_end"
        if is_low_abstraction:
            self.durations = np.ones([source.T, source.height, source.width, inducer.l_height, inducer.l_width])
            self.durations += 0.1*np.random.random([source.T, source.height, source.width, inducer.l_height, inducer.l_width])
        else: # is high abstraction end
            self.durations = np.ones([source.T, source.height, source.width, inducer.h_height, inducer.h_width])
            self.durations += 0.1*np.random.random([source.T, source.height, source.width, inducer.h_height, inducer.h_width])
        self.normalize_paths()
        self.delta_durations = np.zeros_like(self.durations)
        self.min_travel_duration = np.zeros(self.durations.shape[1:])

    def normalize_paths(self):
        sums = self.durations.sum(axis=0, keepdims=True)
        self.durations = self.durations / sums
        return

    def register_update(self, delta_probs):
        self.delta_durations += delta_probs
        return

    def apply_updates(self):
        delta_percent_change = np.minimum(np.maximum(-0.5, self.delta_durations), 2.0)
        delta = self.durations * delta_percent_change
        tmp_durations = delta + self.durations
        sums = tmp_durations.sum(axis=0, keepdims=True)
        rand_init = np.random.random(delta.shape)
        sums_rand = rand_init.sum(axis=0, keepdims=True)
        rand_init = rand_init / sums_rand
        normalized_probs = np.where(sums == 0, rand_init, tmp_durations / sums)
        self.durations = normalized_probs
        return

max_time = 28+28+1
input_pat1 = -np.ones([28, 28])
input_pat1[6:21, 6:21].fill(1.0)
input_pat2 = np.ones([28, 28])
input_pat2[9:18, :].fill(-1.0)
input_pat2[:, 9:18].fill(-1.0)
input_pat1 = (input_pat1 - np.mean(input_pat1))
input_pat1 /= np.max(np.abs(input_pat1))
input_pat2 = (input_pat2 - np.mean(input_pat2))
input_pat2 /= np.max(np.abs(input_pat2))

lab_pat1 = np.concatenate([np.ones([1, 1, 3*3]), -np.ones([1, 1, 3*3])], axis=1)
lab_pat2 = np.concatenate([-np.ones([1, 1, 3*3]), np.ones([1, 1, 3*3])], axis=1)

arr = SourceArray(max_time, 3, 28)
arr.ob_pic_frames(input_pat1, 0, 20)
labs = SeparateSource(max_time, 3, [(14, 9), (14, 18)])
labs.ob_pic_frames(lab_pat1, 0, 25)
inducer = Inducer(9, arr, labs, [(9, 9), (9, 18), (18, 9), (18, 18)])
last_crit_tm = -1
for si in range(10):
    if si % 2 == 0:
        arr.ob_pic_frames(input_pat1, 0, 20)
        labs.ob_pic_frames(lab_pat1, 0, 20)
    else:
        arr.ob_pic_frames(input_pat2, 0, 20)
        labs.ob_pic_frames(lab_pat2, 0, 20)

    local_intensities_whether_critical = []
    for _t in range(1, 40):
        high_ob = inducer.collect_from_high(_t)
        low_ob = inducer.collect_from_low(_t)
        if len(high_ob) == 0 or len(low_ob) == 0:
            continue
        # cri_ct, low_eng, high_eng = inducer.local_mean_field_induction(high_ob, low_ob, _t)
        maintain_critic = (_t > 11)
        cri_ct, low_eng, local_intensities_whether_critical_t = inducer.local_mean_field_induction([], low_ob, _t, maintain_critic=maintain_critic)
        local_intensities_whether_critical.append(local_intensities_whether_critical_t)
        high_eng = "UNK"
        if cri_ct > 0:
            last_crit_tm = _t
        elif last_crit_tm > 0 and cri_ct <= 0 and _t - last_crit_tm >=10:
            break
        print("\t".join([str(_t), str(cri_ct), str(low_eng), str(high_eng)]))

    critical_accumulated_intensity = sum([x[0] for x in local_intensities_whether_critical])
    non_critical_accumulated_intensity = sum([x[1] for x in local_intensities_whether_critical])
    # if si > 0:
    inducer.update_paths()
    print("===============> "+str(si)+"=pat"+str(si % 2 +1) +
          "  intensityRatio="+str(critical_accumulated_intensity)+"/"+str(non_critical_accumulated_intensity))
print("done")

# 1. 正反馈打破对称，(+1,+1)理应比(-1, -1)获得更大的coupling能量。
# 2. 三个阶段：1)无标签预训练low patch的internal energy。2)根据每个low patch的n_bar，计算更新high path使得critic state可以维持。3)带标签训练。