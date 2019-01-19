cd code
python ./run.py --epochs 1000 --k_prior 1
python ./run.py --epochs 1000 --k_prior 2
python ./run.py --epochs 1000 --k_prior 3
python ./run.py --epochs 1000 --k_prior 4

cd ../latex && pdflatex paper.tex && cp paper.pdf ../example/results/;