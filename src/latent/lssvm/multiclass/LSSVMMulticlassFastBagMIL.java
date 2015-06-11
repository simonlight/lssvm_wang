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

public class LSSVMMulticlassFastBagMIL extends LSSVMMulticlassFast<BagMIL,Integer> {

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

	public double testAP(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l, int scale, String simDir, String className) {
             
		List<Evaluation<Integer>> eval = new ArrayList<Evaluation<Integer>>();
        for(int i=0; i<l.size(); i++) {
        	System.out.println(1);
        	Integer y = prediction(l.get(i).input);
        	Integer h = prediction(l.get(i).input.x, y);
        	double score = valueOf(l.get(i).input.x,y,h,w);
        	if (true){
        		File resFile=new File(simDir+"results/metric_"+String.valueOf(scale)+"_"+className+".txt");
        		try {
        			BufferedWriter out = new BufferedWriter(new FileWriter(resFile, true));
        			out.write(Integer.valueOf(y) +","+ Integer.valueOf(h)+","+l.get(i).input.x.getName()+"\n");
        			out.flush();
        			out.close();
        			
        		} catch (IOException e) {
        			// TODO Auto-generated catch block
        			e.printStackTrace();
        		}
        	}
        	eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), (y == 0 ? -1 : 1)*score));
                //System.out.println(l.get(i).label + "\t" + scores[i] + ";");
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}
	

}
