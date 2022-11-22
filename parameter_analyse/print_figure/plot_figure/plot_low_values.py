from parameter_analyse.print_figure.plot_figure.plot import plot_noise_frequency, plot_compare, plot_compare_frequence, plot_compare_all, plot_compare_frequence_noise
import matplotlib.pyplot as plt
# # example of plot
# plot_noise_frequency(6.500000000000001, 50.0, 7.0, 19000.0, 19999.9)
# plot_compare_frequence(0.1, 1.0, 0.0, 19000.0, 19200.9, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0])

# result low amplitude :
plot_compare_frequence(0.30000000000000004, 1.0, 0.0, 10000.0, 20000.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/low_rate_0.png'); plt.close('all')
plot_compare_frequence(0.30000000000000004, 1.0, 7.0, 10000.0, 20000.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/low_rate_7.png'); plt.close('all')
plot_compare_all(0.30000000000000004, 1.0, 0.0, 10000.0, 20000.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
             max_E=5.0)
plt.savefig('figure/noise_low_rate_0.png'); plt.close('all')
plot_compare_all(0.30000000000000004, 1.0, 7.0, 10000.0, 20000.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
                 variance_plot=True)
plt.savefig('figure/noise_low_rate_7.png'); plt.close('all')

# transition
# high pic at the beginning
plot_compare_frequence(0.7000000000000001, 1.0, 0.0, 1000.0, 2000.0, resolution=0.2, window_size=5.0,
             linewidth=[0.5, 2.0], zoom_frequency=100)
plt.savefig('figure/high_pic_0_begin.png'); plt.close('all')
plot_compare_frequence(1.0, 1.0, 7.0, 00.0, 1000.0, resolution=0.2, window_size=5.0,
             linewidth=[0.5, 2.0], zoom_frequency=100)
plt.savefig('figure/high_pic_7_begin.png'); plt.close('all')
plot_compare_all(0.7000000000000001, 1.0, 0.0, 0.0, 1400.0, resolution=0.2, window_size=5.0, path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
                 linewidth=[0.5, 2.0])
plt.savefig('figure/noise_high_pic_0_begin.png'); plt.close('all')
plot_compare_all(1.0, 1.0, 7.0, 00.0, 1000.0, resolution=0.2, window_size=5.0, path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
             linewidth=[0.5, 2.0])
plt.savefig('figure/noise_high_pic_7_begin.png'); plt.close('all')

# variance
plot_compare(0.7000000000000001, 1.0, 0.0, 12000.0, 18000.0, resolution=0.2, window_size=5.0,
             linewidth=[0.5, 2.0], variance_plot=True)
plt.savefig('figure/high_pic_0_end.png'); plt.close('all')
plot_compare(1.0, 1.0, 7.0, 12000.0, 18000.0, resolution=0.2, window_size=5.0,
             linewidth=[0.5, 2.0], variance_plot=True)
plt.savefig('figure/high_pic_7_end.png'); plt.close('all')
plot_compare_all(0.7000000000000001, 1.0, 0.0, 12000.0, 18000.0, resolution=0.2, window_size=5.0, path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
             linewidth=[0.5, 2.0], variance_plot=True, max_E=40.0, min_E=-1.0)
plt.savefig('figure/noise_high_pic_0_end.png'); plt.close('all')
plot_compare_all(1.0, 1.0, 7.0, 12000.0, 18000.0, resolution=0.2, window_size=5.0, path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
             linewidth=[0.5, 2.0], variance_plot=True)
plt.savefig('figure/noise_high_pic_7_end.png'); plt.close('all')

# variance missing
plot_compare_frequence(1.0, 1.0, 0.0, 000.0, 3000.0, resolution=0.2, window_size=5.0,
             linewidth=[0.5, 2.0], zoom_frequency=100, variance_plot=True)
plt.savefig('figure/variance_0_begin.png'); plt.close('all')
plot_compare_frequence(1.0, 1.0, 0.0, 15000.0, 16000.0, resolution=0.2, window_size=5.0,
             linewidth=[0.5, 2.0], zoom_frequency=100, variance_plot=True)
plt.savefig('figure/variance_0_end.png'); plt.close('all')
plot_compare_frequence(1.0, 1.0, 7.0, 000.0, 3000.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/variance_7_begin.png'); plt.close('all')
plot_compare_frequence(1.0, 1.0, 7.0, 15000.0, 16000.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/variance_7_end.png'); plt.close('all')

plot_compare_all(1.0, 1.0, 0.0, 000.0, 3000.0, resolution=0.2, window_size=5.0, path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
             linewidth=[0.5, 2.0], variance_plot=True, max_E=50.0, min_E=-1.0)
plt.savefig('figure/noise_variance_0_begin.png'); plt.close('all')
plot_compare_all(1.0, 1.0, 0.0, 15000.0, 16000.0, resolution=0.2, window_size=5.0, path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
             linewidth=[0.5, 2.0], variance_plot=True, max_E=30.0, min_E=-1.0)
plt.savefig('figure/noise_variance_0_end.png'); plt.close('all')


# high amplitude not reflect in the values
plot_compare_frequence(1.0, 20.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_amplitude_n_0.png'); plt.close('all')
plot_compare_frequence(6.0, 20.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_amplitude_y_0.png'); plt.close('all')
plot_compare_frequence(1.0, 20.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_amplitude_n_7.png'); plt.close('all')
plot_compare_frequence(6.0, 20.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_amplitude_y_7.png'); plt.close('all')
plot_compare_frequence_noise(1.0, 20.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[0.5, 1.0], zoom_frequency=100,
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'], )
plt.savefig('figure/noise_h_amplitude_n_0.png'); plt.close('all')
plot_compare_frequence_noise(6.0, 20.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[0.5, 1.0], zoom_frequency=100,
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],)
plt.savefig('figure/noise_h_amplitude_y_0.png'); plt.close('all')
plot_compare_frequence_noise(1.0, 20.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[0.5, 1.0], zoom_frequency=100,
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
                             variance_plot=True)
plt.savefig('figure/noise_h_amplitude_n_7.png'); plt.close('all')
plot_compare_frequence_noise(6.0, 20.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[0.5, 1.0], zoom_frequency=100,
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],
                             variance_plot=True, max_E=80.0, min_E=-1.0)
plt.savefig('figure/noise_h_amplitude_y_7.png'); plt.close('all')


# high amplitude not reflect in the values
plot_compare_frequence(1.0, 35.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_2_amplitude_n_0.png'); plt.close('all')
plot_compare_frequence(6.0, 35.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_2_amplitude_y_0.png'); plt.close('all')
plot_compare_frequence(1.0, 35.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_2_amplitude_n_7.png'); plt.close('all')
plot_compare_frequence(6.0, 35.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_2_amplitude_y_7.png'); plt.close('all')
plot_compare_frequence_noise(1.0, 35.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'], )
plt.savefig('figure/noise_h_2_amplitude_n_0.png'); plt.close('all')
plot_compare_frequence_noise(6.0, 35.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],)
plt.savefig('figure/noise_h_2_amplitude_y_0.png'); plt.close('all')
plot_compare_frequence_noise(1.0, 35.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'], )
plt.savefig('figure/noise_h_2_amplitude_n_7.png'); plt.close('all')
plot_compare_frequence_noise(6.0, 35.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],)
plt.savefig('figure/noise_h_2_amplitude_y_7.png'); plt.close('all')

# high amplitude not reflect in the values
plot_compare_frequence(1.0, 50.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_3_amplitude_n_0.png'); plt.close('all')
plot_compare_frequence(6.0, 50.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_3_amplitude_y_0.png'); plt.close('all')
plot_compare_frequence(1.0, 50.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_3_amplitude_n_7.png'); plt.close('all')
plot_compare_frequence(6.0, 50.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,
             linewidth=[1.0, 1.0], zoom_frequency=100)
plt.savefig('figure/h_3_amplitude_y_7.png'); plt.close('all')
plot_compare_frequence_noise(1.0, 50.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'], )
plt.savefig('figure/noise_h_3_amplitude_n_0.png'); plt.close('all')
plot_compare_frequence_noise(6.0, 50.0, 0.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],)
plt.savefig('figure/noise_h_3_amplitude_y_0.png'); plt.close('all')
plot_compare_frequence_noise(1.0, 50.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0, linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'], )
plt.savefig('figure/noise_h_3_amplitude_n_7.png'); plt.close('all')
plot_compare_frequence_noise(6.0, 50.0, 7.0, 15000.0, 15500.0, resolution=0.2, window_size=5.0,linewidth=[1.0, 1.0],
                             path_mean_fields=['deterministe', 'stochastic_1e-09', 'stochastic_1e-08'],)
plt.savefig('figure/noise_h_3_amplitude_y_7.png'); plt.close('all')


plt.show()
