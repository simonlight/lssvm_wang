/**
 * 
 */
package lsvm;

import java.util.List;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVMGradientDescentBag extends LSVMGradientDescent<BagMIL,Integer> {

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<LatentRepresentation<BagMIL, Integer>> copy()
			throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected double[] psi(BagMIL x, Integer h) {
		return x.getFeature(h);
	}

	@Override
	protected void init(List<TrainingSample<LatentRepresentation<BagMIL, Integer>>> l) {
		dim = l.get(0).sample.x.getFeature(0).length;
		for(TrainingSample<LatentRepresentation<BagMIL, Integer>> ts : l) {
			ts.sample.h = 0;
		}
	}

	@Override
	protected Integer optimizeH(BagMIL x) {
		int hp = -1;
		double maxVal = -Double.MAX_VALUE;
		/////////////////////????????????????///////////
		for(int i=0; i<x.getInstances().size(); i++) {
			double val = valueOf(x,i);
			if(val > maxVal) {
				maxVal = val;
				hp = i;
			}
		}
		return hp;
	}

}
