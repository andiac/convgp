python myexp.py -k rbf -M 750 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist-raw.test
python myexp.py -k rbf -M 750 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_motion_blur-raw.test
python myexp.py -k rbf -M 750 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_awgn-raw.test
python myexp.py -k rbf -M 750 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_reduced_contrast_and_awgn-raw.test
python myexp.py -k rbf -M 100 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist-dropout_2.test
python myexp.py -k rbf -M 100 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_motion_blur-dropout_2.test
python myexp.py -k rbf -M 100 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_awgn-dropout_2.test
python myexp.py -k rbf -M 100 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_reduced_contrast_and_awgn-dropout_2.test
python myexp.py -k rbf -M 10 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist-dense_2.test
python myexp.py -k rbf -M 10 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_motion_blur-dense_2.test
python myexp.py -k rbf -M 10 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_awgn-dense_2.test
python myexp.py -k rbf -M 10 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_reduced_contrast_and_awgn-dense_2.test
python myexp.py -k rbf -M 1000 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist-flatten_1.test
python myexp.py -k rbf -M 1000 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_motion_blur-flatten_1.test
python myexp.py -k rbf -M 1000 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_awgn-flatten_1.test
python myexp.py -k rbf -M 1000 --learning-rate-block-iters=50000 --learning-rate "0.001 * 10**-(i // b / 3)" --minibatch-size 200 --file myres/mnist_with_reduced_contrast_and_awgn-flatten_1.test
