package latent.lssvm.multiclass;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.stream.ImageInputStream;

import com.sun.org.apache.xml.internal.utils.IntVector;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import sun.font.ExtendedTextLabel;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;

public class LSSVMMulticlassFastBagMILET_negative_positive extends LSSVMMulticlassFastET_negative_positive<BagMIL,Integer> {

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

	protected String[] getETLossPath(BagMIL x, Integer h){
		String featurePath[] = x.getFileFeature(h).split("/");
		String ETLossFileName = featurePath[featurePath.length - 1];
		String imageFileName[] = x.getName().split("/");
		String imClass = imageFileName[imageFileName.length - 1].split("_")[0];
		String root = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
		String negative_ETLossPath =  root + "negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
		String positive_ETLossPath =  root + "ETLoss_ratio/"+ imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
		String[] ETLossPath = {positive_ETLossPath, negative_ETLossPath};
		return ETLossPath;
	}
	
	protected  double getNegative_ETLoss(BagMIL x, Integer h){
		String negative_ETLossPath = getETLossPath(x, h)[1];
		System.out.println(negative_ETLossPath);
		double negativeLoss = negativeLossMap.get(negative_ETLossPath);
		return negativeLoss;
	}
	
	protected  double getPositive_ETLoss(BagMIL x, Integer h){
		String positive_ETLossPath = getETLossPath(x, h)[0];
		double positiveLoss = positiveLossMap.get(positive_ETLossPath);
		return positiveLoss;
	}
	
	protected double delta(Integer yi, Integer yp, BagMIL x, Integer h)  {
//		String featurePath[] = x.getFileFeature(h).split("/");
//		String ETLossFileName = featurePath[featurePath.length - 1];
//		String imageFileName[] = x.getName().split("/");
//		String imClass = imageFileName[imageFileName.length - 1].split("_")[0];
//		String root = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
//		String positive_ETLossPath =  root + "ETLoss_ratio/"+ imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
//		String negative_ETLossPath =  root + "negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
//		System.out.println("-------");
//		System.out.println(h);
//		System.out.println(x.getFileFeature(h));
//		System.out.println(x.getName());
//		System.out.println(x.getFeatures().size());
//		System.out.println(imClass);
//		System.out.println(positive_ETLossPath);
//		System.out.println(negative_ETLossPath);
		String positive_ETLossPath = getETLossPath(x, h)[0];
		String negative_ETLossPath = getETLossPath(x, h)[1];
		double positiveLoss = positiveLossMap.get(positive_ETLossPath);
		double negativeLoss = negativeLossMap.get(negative_ETLossPath);
		
		if(yi == 1 && yp == 1) {
				return (double)(0 + tradeoff*positiveLoss);
			}
		else if (yi == 0 && yp == 0){
				return (double)(0 + tradeoff*negativeLoss);
			}
		else{
				return (double)(1 + tradeoff);
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
		for(int i=0; i<l.size(); i++) {
        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
        	Integer y = prediction(l.get(i).input);
        	Integer h = prediction(l.get(i).input.x, y);
        		File resFile=new File(simDir+"results_neg_pos_new_prediction/metric_"+String.valueOf(scale)+"_"+className+"_"+String.valueOf(tradeoff)+"_"+"pos_neg"+".txt");
        		resFile.getAbsoluteFile().getParentFile().mkdirs();
        		try {
        			BufferedWriter out = new BufferedWriter(new FileWriter(resFile, true));
        			out.write(Integer.valueOf(y) +","+ Integer.valueOf(h)+","+l.get(i).input.x.getName()+"\n");
        			out.flush();
        			out.close();
        			
        		} catch (IOException e) {
        			// TODO Auto-generated catch block
        			e.printStackTrace();
        		}
        	
        	double score = valueOf(l.get(i).input.x,y,h,w);
                
        	eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), (y == 0 ? -1 : 1)*score));
                //System.out.println(l.get(i).label + "\t" + scores[i] + ";");
        }
        double ap = AveragePrecision.getAP(eval);
        return ap;
	}
}
