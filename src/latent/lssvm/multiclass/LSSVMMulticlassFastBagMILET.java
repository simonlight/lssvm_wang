package latent.lssvm.multiclass;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import data.uiuc.mac.LSSVMMulticlassTestET;
import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;

public class LSSVMMulticlassFastBagMILET extends LSSVMMulticlassFastET<BagMIL,Integer> {

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
		if(yi == 1 && yp == 1) {
			return (double)((yi^yp)+tradeoff*(1-gaze_ratio));
			System.out.println(ETLossFileName);
			System.out.println(1-gaze_ratio);
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
//        		File resFile=new File(simDir+"results_3/metric_"+String.valueOf(scale)+"_"+className+"_"+String.valueOf(tradeoff)+"_"+"pos_neg"+".txt");
//        		try {
//        			BufferedWriter out = new BufferedWriter(new FileWriter(resFile, true));
//        			out.write(Integer.valueOf(y) +","+ Integer.valueOf(h)+","+l.get(i).input.x.getName()+"\n");
//        			out.flush();
//        			out.close();
//        			
//        		} catch (IOException e) {
//        			// TODO Auto-generated catch block
//        			e.printStackTrace();
//        		}
        	
        	double score = valueOf(l.get(i).input.x,y,h,w);
                
        	eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), (y == 0 ? -1 : 1)*score));
                //System.out.println(l.get(i).label + "\t" + scores[i] + ";");
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}
	public double testAPRegion(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l, int scale, String simDir, String className, double tradeoff) {
		
		List<Evaluation<Integer>> eval = new ArrayList<Evaluation<Integer>>();
		File resFile=new File(simDir+"std_et/metric_"+String.valueOf(scale)+"_"+className+"_"+String.valueOf(tradeoff)+"_"+"pos_neg"+".txt");
		resFile.getAbsoluteFile().getParentFile().mkdirs();
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(resFile));
			for(int i=0; i<l.size(); i++) {
	        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
	        	Integer y = prediction(l.get(i).input);
	        	Integer h = prediction(l.get(i).input.x, y);	
				out.write(Integer.valueOf(y) +","+ Integer.valueOf(h)+","+l.get(i).input.x.getName()+"\n");
				out.flush();
	        	double score = valueOf(l.get(i).input.x,y,h,w);
	        	eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), (y == 0 ? -1 : 1)*score));
	                //System.out.println(l.get(i).label + "\t" + scores[i] + ";");
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
