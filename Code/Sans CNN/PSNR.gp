# Script Gnuplot pour visualiser l'évolution du PSNR
set terminal pngcairo enhanced font 'arial,12'
set output 'psnr_plot.png'

# Titres et labels
set title "Évolution du PSNR au fil des étapes"
set xlabel "Étapes"
set ylabel "PSNR (dB)"
set grid

# Plot des données depuis le fichier "psnr_values.dat"
# Le fichier doit contenir deux colonnes : l'étape et le PSNR
plot 'psnr_values.dat' using 1:2 with linespoints title 'PSNR', \
     '' using 1:2:(0) with labels offset char 0.5,1 notitle
