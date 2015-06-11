package latent.lssvm.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;

public class LSSVMMulticlassBagMIL extends LSSVMMulticlass<BagMIL,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2238562266157178176L;

	@Override
	protected List<Integer> enumerateH(BagMIL x) {
		List<Integer> latent = new ArrayList<Integer>();
		for(int i=0; i<x.getFeatures().size(); i++) {
			latent.add(i);
		}
		return latent;
	}

	@Override
	protected double[] psi(BagMIL x, Integer y, Integer h) {
		double[] psi = new double[dim];
		for(int i=0; i<x.getFeature(h).length; i++) {
			psi[y*x.getFeature(h).length+i] = x.getFeature(h)[i];
		}
		return psi;
	}

	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l) {
		
		int nbClass = 0;
		for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;
		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}
		double[] nb = new double[nbClass];
		for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : l) {
			nb[ts.output]++;
		}
		
		dim = listClass.size() * l.get(0).input.x.getFeature(0).length;
		System.out.println("Multiclass \t dim= " + dim + " \tclasses: " + listClass + "\t" + Arrays.toString(nb));
	}
	
	public double test(List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> l) {
		double[] nb = new double[listClass.size()];
		for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : l) {
			nb[ts.output]++;
		}
		System.out.println("Test - class: " + listClass + "\t" + Arrays.toString(nb));
		return accuracy(l);
	}

}
