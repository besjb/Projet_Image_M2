# Script Gnuplot pour visualiser l'évolution du SSIM
set terminal pngcairo enhanced font 'arial,12'
set output 'ssim_plot.png'

# Titres et labels
set title "Évolution du SSIM au fil des étapes"
set xlabel "Étapes"
set ylabel "SSIM"
set grid

# Plot des données depuis le fichier "ssim_values.dat"
# Le fichier doit contenir deux colonnes : l'étape et le SSIM
plot 'ssim_values.dat' using 1:2 with linespoints title 'SSIM', \
     '' using 1:2:(0) with labels offset char 0.5,1 notitle
