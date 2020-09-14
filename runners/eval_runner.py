from losses.dsm import dsm_score_evaluation
from .abc_runner import *

__all__ = ['EvalRunner']

class EvalRunner(Runner):

    @torch.no_grad()
    def run(self):

        dataloader = self.get_dataloader(bs=self.config.sampling.batch_size)
        score = self.load_score(eval=True)
        sigmas = self.get_sigmas(npy=True)

        hashmap = dict([(i.cpu().item(), [[], [], []]) for i in sigmas])

        for k in range(10):
            for X, _ in dataloader:
                X = X.to(self.args.device)
                X = data_transform(self.config.data, X)
                labels, variance, covariance, l2 = dsm_score_evaluation(self.args, score, X, sigmas)

                for label_i, st0, st1, st2 in zip(*[x.cpu().numpy() for x in [labels, variance, covariance, l2]]):
                    hashmap[label_i][0] += [st0]
                    hashmap[label_i][1] += [st1]
                    hashmap[label_i][2] += [st2]
            print(f"{k} / 10")

        res = []
        for key, value in hashmap.items():
            value = np.array(value)
            res += [[key, [[v.mean(), v.var()] for v in value]]]
            if self.args.verbose.upper() == 'INFO':
                print(res[-1])

        np.save(os.path.join(self.args.log_path, 'sigma_eval.npy'), np.array(res))
