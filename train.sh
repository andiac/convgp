python myexp.py -k rbf -M 750 --learning-rate-block-iters=30000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/raw.train
python myexp.py -k rbf -M 100 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/dropout_2.train
python myexp.py -k rbf -M 10 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/dense_2.train
# python myexp.py -k rbf -M 1000 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/flatten_1.train
