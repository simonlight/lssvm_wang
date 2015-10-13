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

public class LSSVMMulticlassFastBagMILET extends LSSVMMulticlassFastET<BagMIL,Integer> {

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
	
	//50--36, 60--25, 70--16...
	public int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
	
	public Integer getGazeInitRegion(BagMIL x, int scale){
		Integer maxH=-1;
		double maxGazeRatio = -1;
		for (Integer h=0;h<convertScale(scale);h++){
			double gazeRatio = getGazeRatio(x, h);
			if (gazeRatio>=maxGazeRatio){
				maxH=h;
				maxGazeRatio = gazeRatio;
			}
		}
		return maxH;
	}
	
	protected double getGazeRatio(BagMIL x, Integer h){
		String cls = x.getName().split("_")[0];
		String featurePath[] = x.getFileFeature(h).split("/");
		String ETLossFileName = featurePath[featurePath.length - 1];
		System.out.println(cls+"_"+ETLossFileName);
		System.out.println(lossMap);
		double gaze_ratio = lossMap.get(cls+"_"+ETLossFileName);
		return gaze_ratio;
	}
	
	protected double delta(Integer yi, Integer yp, BagMIL x, Integer h)  {
		double gaze_ratio = getGazeRatio(x, h);
//		System.out.println(ETLossFileName);
//		System.out.println(1-gaze_ratio);
		if(yi == 1 && yp == 1) {
//			System.out.println(tradeoff*(1-gaze_ratio));
			return (double)(0+tradeoff*(1-gaze_ratio));
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
