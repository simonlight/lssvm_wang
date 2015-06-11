package latent.lssvm.multiclass;

import java.util.List;

import latent.LatentRepresentation;
import latent.lssvm.LSSVM;
import struct.STrainingSample;

public abstract class LSSVMMulticlass<X,H> extends LSSVM<X,Integer,H> {
	
	protected List<Integer> listClass = null;
	
	protected abstract List<H> enumerateH(X x);

	@Override
	public Integer prediction(LatentRepresentation<X, H> lr) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			for(H h : enumerateH(lr.x)) {
				double val = valueOf(lr.x,y,h,w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
				}
			}
		}
		return ypredict;
	}

	@Override
	protected double delta(Integer yi, Integer yp) {
		if(yi == yp) {
			return 0;
		}
		else {
			return 1;
		}
	}

	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X, H>, Integer> ts) {
		int ypredict = -1;
		H hpredict = null;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			for(H h : enumerateH(ts.input.x)) {
				double val = delta(ts.output, y) + valueOf(ts.input.x,y,h,w);
				if(val>valmax){
					valmax = val;
					ypredict = y;
					hpredict = h;
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = ypredict;
		res[1] = hpredict;
		return res;
	}

	@Override
	protected H prediction(X x, Integer y) {
		H hpredict = null;
		double valmax = -Double.MAX_VALUE;
		for(H h : enumerateH(x)) {
			double val = valueOf(x,y,h,w);
			if(val>valmax){
				valmax = val;
				hpredict = h;
			}
		}
		return hpredict;
	}
	
	protected double valueOf(X x, Integer y, H h, double[] w) {
		// <w, Psi(x,y)>
		return linear.valueOf(w, psi(x,y,h));
	}
	
	protected double accuracy(List<STrainingSample<LatentRepresentation<X, H>, Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

}
