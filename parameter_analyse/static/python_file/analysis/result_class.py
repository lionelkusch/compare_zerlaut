#  Copyright 2023 Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "
import numpy as np


class Result_analyse:
    def __init__(self, max_lag_autocorrelation=50):
        self.max_lag_autocorrelation = max_lag_autocorrelation
        self.data = {}
        list_single = ['names_population',
                       'synch_Rs_times_init',
                       'synch_Rs_times_end', 'percentage',
                       'percentage_burst', 'percentage_burst_cv', 'cvs_IFR_0_1ms', 'cvs_IFR_1ms',
                       'frequency_hist_1_freq', 'frequency_hist_1_val', 'frequency_phase_freq', 'frequency_phase_val',
                       ]
        list_mean = ['cvs_ISI', 'lvs_ISI',
                     'burst_cv_begin', 'burst_lv_begin', 'burst_cv_end', 'burst_lv_end'
                     ]
        list_mean_max = ['rates', 'ISI',
                         'synch_Rs',
                         'burst_nb', 'burst_count', 'burst_rate', 'burst_interval'
                         ]
        for i in list_single:
            self.data[i] = []
        for i in list_mean:
            self.data[i + '_average'] = []
            self.data[i + '_std'] = []
        for i in list_mean_max:
            self.data[i + '_average'] = []
            self.data[i + '_std'] = []
            self.data[i + '_max'] = []
            self.data[i + '_min'] = []

    def empty(self):
        for key in self.data.keys():
            self.data[key].append(None)

    def set_name_population(self, name_population):
        self.data['names_population'] = name_population

    def save_name_population(self, name_population):
        self.data['names_population'].append(name_population)

    def save_rate(self, rate):
        self._save_mean_max('rates', rate)

    def save_simple_synchronization(self,
                                    cvs_IFR_0_1ms,
                                    cvs_IFR_1ms,
                                    ):
        self.data['cvs_IFR_0_1ms'].append(cvs_IFR_0_1ms)
        self.data['cvs_IFR_1ms'].append(cvs_IFR_1ms)

    def save_irregularity(self, cvs_ISI, lvs_ISI):
        self._save_mean('cvs_ISI', cvs_ISI)
        self._save_mean('lvs_ISI', lvs_ISI)

    def save_ISI(self, ISI):
        self._save_mean_max('ISI', ISI)

    def save_R_synchronization(self, R_synch, synch_Rs_times_init, synch_Rs_times_end):
        self._save_mean_max('synch_Rs', R_synch)
        self.data['synch_Rs_times_init'].append(synch_Rs_times_init)
        self.data['synch_Rs_times_end'].append(synch_Rs_times_end)

    def save_percentage(self, percentage):
        self.data['percentage'].append(percentage)

    def save_burst_nb(self, nb_burst):
        self._save_mean_max('burst_nb', nb_burst)

    def save_burst_count(self, count_burst):
        self._save_mean_max('burst_count', count_burst)

    def save_burst_rate(self, rate):
        self._save_mean_max('burst_rate', rate)

    def save_burst_interval(self, interval):
        self._save_mean_max('burst_interval', interval)

    def save_burst_begin_irregularity(self, cvs_ISI, lvs_ISI):
        self._save_mean('burst_cv_begin', cvs_ISI)
        self._save_mean('burst_lv_begin', lvs_ISI)

    def save_burst_end_irregularity(self, cvs_ISI, lvs_ISI):
        self._save_mean('burst_cv_end', cvs_ISI)
        self._save_mean('burst_lv_end', lvs_ISI)

    def save_burst_percentage(self, percentage):
        self.data['percentage_burst'].append(percentage)

    def save_burst_percentage_cv(self, percentage_cv):
        self.data['percentage_burst_cv'].append(percentage_cv)

    def save_frequency_hist_1(self, frequency_hist_1):
        self.data['frequency_hist_1_freq'].append(frequency_hist_1[0])
        self.data['frequency_hist_1_val'].append(frequency_hist_1[1])

    def save_frequency_phase(self, frequency_phase):
        self.data['frequency_phase_freq'].append(frequency_phase[0])
        self.data['frequency_phase_val'].append(frequency_phase[1])

    def _save_mean(self, name, value, mask=None):
        if type(value) == list and value == []:
            self.data[name + '_average'].append(None)
            self.data[name + '_std'].append(None)
        else:
            if type(value[0]) == np.ndarray:
                value = np.concatenate(value)
            if type(value) == np.ndarray and value.shape[0] == 0:
                self.data[name + '_average'].append(None)
                self.data[name + '_std'].append(None)
                self.data[name + '_max'].append(None)
                self.data[name + '_min'].append(None)
                return
            if mask is not None:
                value = np.ma.masked_array(value, mask=mask)
            self.data[name + '_average'].append(float(np.mean(value)))
            self.data[name + '_std'].append(float(np.std(value)))

    def _save_mean_max(self, name, value, mask=None):
        if type(value) == list and value == []:
            self.data[name + '_average'].append(None)
            self.data[name + '_std'].append(None)
            self.data[name + '_max'].append(None)
            self.data[name + '_min'].append(None)
        else:
            if type(value[0]) == np.ndarray:
                value = np.concatenate(value)
            if type(value) == np.ndarray and value.shape[0] == 0:
                self.data[name + '_average'].append(None)
                self.data[name + '_std'].append(None)
                self.data[name + '_max'].append(None)
                self.data[name + '_min'].append(None)
                return
            if mask is not None:
                value = np.ma.masked_array(value, mask=mask)
            self.data[name + '_average'].append(float(np.mean(value)))
            self.data[name + '_std'].append(float(np.std(value)))
            self.data[name + '_max'].append(float(np.max(value)))
            self.data[name + '_min'].append(float(np.min(value)))

    def result(self):
        return self.data

    def name_measure(self):
        return self.data.keys()

    def print_result(self):
        print('Name population: %r ' % self.data['names_population'])
        print('Mean Rates: %r Hz' % self.data['rates_average'])
        print('Standard deviation of rates: %r Hz' % self.data['rates_std'])
        print('Maximum of rates: %r Hz' % self.data['rates_max'])
        print('Minimum of rates: %r Hz' % self.data['rates_min'])
        print('Mean Inter-Spike Interval: %r ms' % self.data['ISI_average'])
        print('Standard deviation of Inter-Spike Interval: %r ms' % self.data['ISI_std'])
        print('Maximum of Inter-Spike Interval: %r ms' % self.data['ISI_max'])
        print('Minimum of Inter-Spike Interval: %r ms' % self.data['ISI_min'])
        print('Mean Cv ISI: %r' % self.data['cvs_ISI_average'])
        print('Standard deviation of Cv ISI: %r' % self.data['cvs_ISI_std'])
        print('Mean lv ISI: %r' % self.data['lvs_ISI_average'])
        print('Standard deviation of lv ISI: %r' % self.data['lvs_ISI_std'])

        print('Variation hist 0.1, 1, 5, w5 ms: %r %r %r %r' % (self.data['cvs_IFR_0_1ms'], self.data['cvs_IFR_1ms'],
                                                                self.data['cvs_IFR_5ms'],self.data['cvs_IFR_w5ms']))
        print('Max hist 0.1, 1, 5, w5 ms: %r  %r %r %r' % (self.data['max_IFR_0_1ms'], self.data['max_IFR_1ms'],
                                                           self.data['max_IFR_5ms'], self.data['max_IFR_w5ms']))
        print('Timescale 0.1, 1, 5, w5 ms: %r  %r %r %r' % (self.data['timescale_0_1ms'], self.data['timescale_1ms'],
                                                            self.data['timescale_5ms'], self.data['timescale_w5ms']))
        print("PLV 0.1, 1, 5, w5 ms: %r %r %r %r" % (self.data['PLV_0_1ms'], self.data['PLV_1ms'],
                                                     self.data['PLV_5ms'], self.data['PLV_w5ms']))
        print("PLV_angle  0.1, 1, 5, w5 ms: %r %r %r %r" % (self.data['PLV_angle_0_1ms'], self.data['PLV_angle_1ms'],
                                                            self.data['PLV_angle_5ms'], self.data['PLV_angle_w5ms']))
        print("Mean Phase Shift  0.1, 1, 5, w5 ms: %r %r %r %r" % (self.data['MeanPhaseShift_0_1ms'], self.data['MeanPhaseShift_1ms'],
                                                                   self.data['MeanPhaseShift_5ms'], self.data['MeanPhaseShift_w5ms']))
        # str_auto = " "
        # for i in range(self.max_lag_autocorrelation):
        #     str_auto += " %r"%(self.data['autocorrelation_1_ms_'+str(i)][0])
        # print('Autocorrelation 1 ms: %s ' % str_auto)

        print('Mean R synchronize : %r ' % self.data['synch_Rs_average'])
        print('Standard deviation of R synchronize : %r ' % self.data['synch_Rs_std'])
        print('Time R init synchronize : %r ' % self.data['synch_Rs_times_init'])
        print('Time R end synchronize : %r ' % self.data['synch_Rs_times_end'])
        print('Percentage :%r' % self.data['percentage'])
        print('Burst mean nb: %r ' % self.data['burst_nb_average'])
        print('Burst standard deviation of nb : %r ' % self.data['burst_nb_std'])
        print('Burst max nb : %r ' % self.data['burst_nb_max'])
        print('Burst min nb : %r ' % self.data['burst_nb_min'])
        print('Burst mean count: %r ' % self.data['burst_count_average'])
        print('Burst standard deviation of count : %r ' % self.data['burst_count_std'])
        print('Burst max count : %r ' % self.data['burst_count_max'])
        print('Burst min count : %r ' % self.data['burst_count_min'])
        print('Burst mean rate: %r Hz' % self.data['burst_rate_average'])
        print('Burst standard deviation of rate : %r Hz' % self.data['burst_rate_std'])
        print('Burst max rate: %r Hz' % self.data['burst_rate_max'])
        print('Burst min rate : %r Hz' % self.data['burst_rate_min'])
        print('Burst mean interval: %r ms' % self.data['burst_interval_average'])
        print('Burst standard deviation of interval : %r ms' % self.data['burst_interval_std'])
        print('Burst max interval: %r ms' % self.data['burst_interval_max'])
        print('Burst min interval : %r ms' % self.data['burst_interval_min'])
        print('Burst mean cv begin time: %r ms' % self.data['burst_cv_begin_average'])
        print('Burst standard deviation of cv begin time: %r ms' % self.data['burst_cv_begin_std'])
        print('Burst mean lv begin time: %r ms' % self.data['burst_lv_begin_average'])
        print('Burst standard deviation of lv begin time: %r ms' % self.data['burst_lv_begin_std'])
        print('Burst mean cv end time: %r ms' % self.data['burst_cv_end_average'])
        print('Burst standard deviation of cv end time: %r ms' % self.data['burst_cv_end_std'])
        print('Burst mean lv end time: %r ms' % self.data['burst_lv_end_average'])
        print('Burst standard deviation of lv end time: %r ms' % self.data['burst_lv_end_std'])
        print('Burst percentage :%r' % self.data['percentage_burst'])
        print('Burst percentage irregularity :%r' % self.data['percentage_burst_cv'])
