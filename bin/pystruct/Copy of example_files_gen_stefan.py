from pystruct.models import DirectionalGridCRF
import pystruct.learners as ssvm
from pystruct.datasets import generate_blocks_multinomial
from pystruct.plot_learning import plot_learning

X, Y = generate_blocks_multinomial(noise=2, n_samples=20, seed=1)
crf = DirectionalGridCRF(inference_method="qpbo", neighborhood=4)
clf = ssvm.OneSlackSSVM(model=crf, n_jobs=-1, inference_cache=100,
                        show_loss_every=10,
                        switch_to=("ad3", {'branch_and_bound': True}))
clf.fit(X, Y)

plot_learning(clf, time=False)