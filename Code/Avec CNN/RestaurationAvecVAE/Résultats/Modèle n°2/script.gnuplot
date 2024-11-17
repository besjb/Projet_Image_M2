set terminal pngcairo size 800,600 enhanced font 'Arial,10'
set output 'epochs_summary.png'

set title "Résumé des Époques"
set xlabel "Époque"
set ylabel "Pertes"
set grid

plot 'training_metrics_B50_E15_Adamax.dat' using 1:2 with lines title 'Perte Totale',      'training_metrics_B50_E15_Adamax.dat' using 1:3 with lines title 'Perte Reconstruction',      'training_metrics_B50_E15_Adamax.dat' using 1:4 with lines title 'Perte KL'
