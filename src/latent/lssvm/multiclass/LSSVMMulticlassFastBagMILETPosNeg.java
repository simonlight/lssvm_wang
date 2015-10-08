package latent.lssvm.multiclass;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;

public class LSSVMMulticlassFastBagMILETPosNeg extends LSSVMMulticlassFastETPosNeg<BagMIL,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7682761029498647460L;

	@Override
	protected List<Integer> enumerateH(BagMIL x) {
		//how many kinds of latent values
		List<Integer> latent = new ArrayList<Integer>();
		for(int i=0; i<x.getFeatures().size(); i++) {
			latent.add(i);
		}
		return latent;
	}

	@Override
	protected double[] psi(BagMIL x, Integer h) {
		//return the list of image features
		return x.getFeature(h);
	}
	
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l) {

		// initialize the one class model dimension
		dim = l.get(0).input.x.getFeature(0).length;
	}

	
	protected double delta(Integer yi, Integer yp, BagMIL x, Integer h)  {
		String featurePath[] = x.getFileFeature(h).split("/");
		String ETLossFileName = featurePath[featurePath.length - 1];
		double gaze_ratio = lossMap.get(ETLossFileName);
		double neg_gaze_ratio = lossMapNeg.get(ETLossFileName);
		
//		System.out.println(ETLossFileName);
//		System.out.println(1-gaze_ratio);
		if(yi == 1 && yp == 1) {
			
			return (double)(0+tradeoff*(1-gaze_ratio));
		}
		else if (yi == 0 && yp == 0){
			return (double)(0+tradeoff*(1-neg_gaze_ratio));
		}
		else {
			return (double)((yi^yp));
		}		
	}

	
	public double testAP(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l) {
		
		List<Evaluation<Integer>> eval = new ArrayList<Evaluation<Integer>>();
		for(int i=0; i<l.size(); i++) {
        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
        	Integer y = prediction(l.get(i).input);
        	Integer h = prediction(l.get(i).input.x, y);
        	double score = valueOf(l.get(i).input.x,y,h,w);
                
        	eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), (y == 0 ? -1 : 1)*score));
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}
	
	
	public double testAPRegion(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l, double epsilon, double lambda,int scale, String simDir, String className, double tradeoff, String detailFolder) {
		
		List<Evaluation<Integer>> eval = new ArrayList<Evaluation<Integer>>();
		File resFile=new File(simDir+detailFolder+"/metric_"+String.valueOf(tradeoff)+"_"+String.valueOf(scale)+"_"+Double.valueOf(epsilon)+"_"+Double.valueOf(lambda)+"_"+className+".txt");

		resFile.getAbsoluteFile().getParentFile().mkdirs();
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
			for(int i=0; i<l.size(); i++) {
	        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
				Integer yp = prediction(l.get(i).input);
	        	Integer h = prediction(l.get(i).input.x, yp);
	        	Integer yi = l.get(i).output;
				out.write(Integer.valueOf(yp) +","+Integer.valueOf(yi) +","+ Integer.valueOf(h)+","+l.get(i).input.x.getName()+"\n");
				out.flush();
	        	double score = valueOf(l.get(i).input.x,yp,h,w);
	        	eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), (yp == 0 ? -1 : 1)*score));
	        }
			out.close();
	        	
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}
}