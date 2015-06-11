package ssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import struct.STrainingSample;

public class SSVMMulticlass extends SSVM<double[], Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2968899551726671188L;
	
	protected List<Integer> listClass = null;

	@Override
	public Integer prediction(double[] x) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = valueOf(x,y,w);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	protected Integer lossAugmentedInference(STrainingSample<double[], Integer> ts, double[] w) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			double val = delta(ts.output,y) + valueOf(ts.input,y,w);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		return ypredict;
	}

	@Override
	public double evaluation(List<STrainingSample<double[], Integer>> l) {
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<double[], Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}

	@Override
	protected double delta(Integer yi, Integer y) {
		if(y == yi) {
			return 0;
		}
		else {
			return 1;
		}
	}

	@Override
	protected double[] psi(double[] x, Integer y) {
		double[] psi = new double[dim];
		for(int i=0; i<x.length; i++) {
			psi[y*x.length+i] = x[i];
		}
		return psi;
	}

	@Override
	protected double valueOf(double[] x, Integer y, double[] w) {
		// <w, Psi(x,y)>
		return linear.valueOf(w, psi(x,y));
	}

	@Override
	protected void init(List<STrainingSample<double[], Integer>> l) {
		int nbClass = 0;
		for(STrainingSample<double[], Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;
		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}
		double[] nb = new double[nbClass];
		for(STrainingSample<double[], Integer> ts : l) {
			nb[ts.output]++;
		}
		System.out.println("SSVM multiclass - classes: " + listClass + "\t" + Arrays.toString(nb));
		dim = l.get(0).input.length*listClass.size();
	}
	
	public String toString() {
		return "ssvm_multiclass_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin;
	}
}
