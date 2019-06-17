# Produce experiments results for CartPole
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 5 -dna --exp_name sb_no_rtg_dna
python plot.py data/
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna
python plot.py data/
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 5 -rtg --exp_name sb_rtg_na
python plot.py data/
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 5 -dna --exp_name lb_no_rtg_dna
python plot.py data/
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna
python plot.py data/
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 5 -rtg --exp_name lb_rtg_na
python plot.py data/
