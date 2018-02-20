"""
myexp.py
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import GPflow
import exp_tools
import opt_tools
import convgp.convkernels as ckern

import pickle

class MnistExperiment(exp_tools.MyMnistExperiment):
    def __init__(self, name=None, M=100, run_settings=None):
        name = "fullmnist-%s%i" % (run_settings['kernel'], M) if name is None else name
        super(MnistExperiment, self).__init__(name)
        self.run_settings = run_settings if run_settings is not None else {}
        self.M = M

    def setup_model(self):
        Z = None
        if self.run_settings['kernel'] == "rbf":
            k = GPflow.kernels.RBF(self.X.shape[1])
            Z = self.X[np.random.permutation(len(self.X))[:self.M], :]
        else:
            raise NotImplementedError

        if Z is None:
            Z = (k.kern_list[0].init_inducing(self.X, self.M, method=self.run_settings['Zinit'])
                 if type(k) is GPflow.kernels.Add else
                 k.init_inducing(self.X, self.M, method=self.run_settings['Zinit']))

        k.fixed = self.run_settings.get('fixed', False)

        self.m = GPflow.svgp.SVGP(self.X, self.Y, k, GPflow.likelihoods.MultiClass(10), Z.copy(), num_latent=10,
                                  minibatch_size=self.run_settings.get('minibatch_size', self.M))
        if self.run_settings["fix_w"]:
            self.m.kern.W.fixed = True

    def setup_logger(self, verbose=False):
        h = pd.read_pickle(self.hist_path) if os.path.exists(self.hist_path) else None
        if h is not None:
            print("Resuming from %s..." % self.hist_path)
        tasks = [
            opt_tools.tasks.DisplayOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            opt_tools.tasks.GPflowLogOptimisation(opt_tools.seq_exp_lin(1.1, 20)),
            exp_tools.GPflowMultiClassificationTrackerLml(
                self.Xt[:, :], self.Yt[:, :], itertools.count(1800, 1800), trigger="time",
                verbose=True, store_x="final_only", store_x_columns=".*(variance|lengthscales)"),
            opt_tools.gpflow_tasks.GPflowMultiClassificationTracker(
                self.Xt[:, :], self.Yt[:, :], opt_tools.seq_exp_lin(1.5, 150, start_jump=30), trigger="time",
                verbose=True, store_x="final_only", store_x_columns=".*(variance|lengthscales)", old_hist=h),
            opt_tools.tasks.StoreOptimisationHistory(self.hist_path, opt_tools.seq_exp_lin(1.5, 600, start_jump=30),
                                                     trigger="time", verbose=False),
            opt_tools.tasks.Timeout(self.run_settings.get("timeout", np.inf))
        ]
        self.logger = opt_tools.GPflowOptimisationHelper(self.m, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MNIST experiment.')
    parser.add_argument('--fixed', '-f', help="Fix the model hyperparameters.", action="store_true", default=False)
    parser.add_argument('--name', '-n', help="Experiment name appendage.", type=str, default=None)
    parser.add_argument('--learning-rate', '-l', help="Learning rate.", type=str, default="0.001")
    parser.add_argument('--learning-rate-block-iters', type=int, default=50000,
                        help="How many iterations to use in a run with a single learning rate.")
    parser.add_argument('--profile', help="Only run a quick profile of an iteration.", action="store_true",
                        default=False)
    parser.add_argument('--no-opt', help="Do not optimise.", action="store_true", default=False)
    parser.add_argument('-M', help="Number of inducing points.", type=int, default=100)
    parser.add_argument('--minibatch-size', help="Size of the minibatch.", type=int, default=100)
    parser.add_argument('--benchmarks', action="store_true", default=False)
    parser.add_argument('--optimiser', '-o', type=str, default="adam")
    parser.add_argument('--fix-w', action="store_true", default=False)
    parser.add_argument('--kernel', '-k', help="Kernel.")
    parser.add_argument('--Zinit', help="Inducing patches init.", default="patches-unique", type=str)
    parser.add_argument('--lml', help="Compute log marginal likelihood.", default=False, action="store_true")
    parser.add_argument('--file', type=str, default=None)
    args = parser.parse_args()

    # if GPflow.settings.dtypes.float_type is not tf.float32:
    #     raise RuntimeError("float_type must be float32, as set in gpflowrc.")

    run_settings = vars(args).copy()
    del run_settings['profile']
    del run_settings['no_opt']
    del run_settings['name']

    if args.file:
        mydict = pickle.load(open(args.file, "rb"))
        line = args.file.split('.')
        print(line)

        if line[1] == 'train':
            exp = MnistExperiment(name=line[0].split('/')[1], M=args.M, run_settings=run_settings)
            print("loading mydict, shape: ", mydict["X"].shape, mydict["Y"].shape)
            exp.X = mydict["X"]
            exp.Y = mydict["Y"].reshape(-1, 1)
            exp.Xt = mydict["Xt"]
            exp.Yt = mydict["Yt"].reshape(-1, 1)
            i = pd.read_pickle(exp.hist_path).i.max() if os.path.exists(exp.hist_path) else 1.0
            b = args.learning_rate_block_iters
            print("learning rate: %s" % args.learning_rate)
            run_settings['learning_rate'] = eval(args.learning_rate)  # Can use i and b in learning_rate
            print(run_settings['learning_rate'], i)
            exp.setup()
            rndstate = np.random.randint(0, 1e9)
            exp.m.X.index_manager.rng = np.random.RandomState(rndstate)
            exp.m.Y.index_manager.rng = np.random.RandomState(rndstate)
            exp.run(maxiter=args.learning_rate_block_iters)

        else:
            exp = MnistExperiment(name=line[0].split('-')[1], M=args.M, run_settings=run_settings)
            
            exp.X = mydict["feature"]
            exp.Y = mydict["Yt"].reshape(-1, 1)
            exp.Xt = mydict["feature"]
            exp.Yt = mydict["Yt"].reshape(-1, 1)

            exp.setup()
            p, var = exp.m.predict_y(exp.Xt)
            lpp = np.mean(np.log(p[np.arange(len(exp.Xt)), exp.Yt.flatten()]))
            acc = np.mean(p.argmax(1) == exp.Yt[:, 0])

            print("Accuracy: %f" % acc)
            print("Lpp     : %f" % lpp)

            mydict["accuracy"] = acc
            mydict["prob"] = p
            mydict["var"] = var

            pickle.dump(mydict, open(args.file + ".res", "wb"))
