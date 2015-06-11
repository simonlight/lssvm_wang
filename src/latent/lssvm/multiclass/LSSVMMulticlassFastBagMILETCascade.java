package latent.lssvm.multiclass;

import java.awt.Point;
import java.awt.Rectangle;
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
import javax.imageio.stream.ImageInputStream;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;

public class LSSVMMulticlassFastBagMILETCascade extends LSSVMMulticlassFastCascade<BagMIL,Integer> {

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
	
	protected Object[] lossAugmentedInference2(STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer> ts) {
		int ypredict = -1;
		Integer hpredict = null;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			for(Integer h : enumerateH(ts.input.x)) {
//				double val = delta2(ts.output, y, ts.input.x, h) + valueOf(ts.input.x, ts.output, h, w);
				double val = delta2(ts.output, y, ts.input.x, h) + valueOf(ts.input.x, y, h, w);
//				System.out.println(valueOf(ts.input.x, ts.output, h, w));
//				double val = delta2(ts.output, y, ts.input.x, h) + valueOf(ts.input.x, y, h, w);
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
	
	protected Integer prediction2(BagMIL x) {
		
		Integer hpredict = null;
		double fixations_rate = 0;
		for(Integer h : enumerateH(x)) {
			String[] paths = BagMILPath2ETGazesPath(x,h);
			String ETGazesPath = paths[0];
			String ETLossPath = paths[1];
			File f = new File(ETLossPath);
			f.getAbsoluteFile().getParentFile().mkdirs();
			if(!f.exists()){
				//double ETLoss = getETIntersectionLoss(x, h, ETBBPath);
				double ETLoss = getETLossRatio(x, h, ETGazesPath);
				if (1 - ETLoss > fixations_rate){
					hpredict = h;	
					fixations_rate = 1 - ETLoss;
				}
			}
			else{
				try {
					FileReader fr = new FileReader(ETLossPath);
					BufferedReader br=new BufferedReader(fr);
					double ETLoss = Double.parseDouble(br.readLine());
					if (1 - ETLoss > fixations_rate){
						hpredict = h;										
						fixations_rate = 1 - ETLoss;
					}
					br.close();
					fr.close();	
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}	
			}
		}
		//test pass, always select h with highest fixations_rate
		return hpredict;
	}
	
	protected double delta2(Integer yi, Integer yp, BagMIL x, Integer h)  {
		String[] paths = BagMILPath2ETGazesPath(x,h);
		String ETGazesPath = paths[0];
		String ETLossPath = paths[1];
		File f = new File(ETLossPath);
		double globalRatio=1.0;
		f.getAbsoluteFile().getParentFile().mkdirs();
		if(!f.exists()) { 	
			//double ETLoss = getETIntersectionLoss(x, h, ETBBPath);
			double ETLoss = getETLossRatio(x, h, ETGazesPath);
			try {
				BufferedWriter out = new BufferedWriter(new FileWriter(f));
				out.write(String.valueOf(ETLoss));
				out.flush();
				out.close();
				if(yi == 1 && yp==1) {
					//System.out.println(x.getName() + "\t" + (0+ETLoss));
					
					return (double)(globalRatio*ETLoss);
//					return (double)(ETLoss);
					//return (double)(0+ETLoss);
					}
				else {
					//System.out.println(x.getName() + "\t" + (1+ETLoss));
					return (double)(yi^yp);
				}
			} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.err.println("No ETLoss file");
			return -10000;
			}
		}
		else{		
			try {
				FileReader fr = new FileReader(ETLossPath);
				BufferedReader br=new BufferedReader(fr);
				double ETLoss = Double.parseDouble(br.readLine());
				br.close();
				fr.close();
				if(yi == 1 && yp==1) {
					//System.out.println(x.getName() + "\t" + (0+ETLoss));
//					return (double)(0+0);
					//checked normal
					return (double)(globalRatio*ETLoss);
//					return (double)(ETLoss);
					//return (double)(0+ETLoss);
				}
				else {
					//System.out.println(x.getName() + "\t" + (1+ETLoss));
//					return (double)(1+0);
					return (double)(yi^yp);
//					return (double)(1);
				}
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.err.println("No ETLoss file");
				return -10000;
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.err.println("No ETLoss file");
				return -10000;
				
			}	
		}
	}
	///////////////////////////
	protected int[] getPatch(Integer totalPatchNum, Integer h){
		Double root = Math.sqrt(totalPatchNum);
		Integer denominator = root.intValue();
		int patch[] = {h/denominator, h%denominator};
		return patch;
	}
	
	protected Rectangle getPatchBB(int width, int height, int totalPatchNum, int[] patch){
		int up = (int)(1 + Math.floor(patch[0]*0.1*height));
		int down = (int)Math.floor(up + height*(1-(Math.sqrt(totalPatchNum)-1)/10));
		int left = (int)(1 + Math.floor(patch[1]*0.1*width));
		int right = (int)Math.floor(left + width*(1-(Math.sqrt(totalPatchNum)-1)/10));
		return new Rectangle(left,up,right-left,down-up);
	}
	
	protected ArrayList<Point> getGazes(String ETGazesPath){
		
		ArrayList<Point> gazes = new ArrayList<Point>();
		try {
			FileReader fr = new FileReader(ETGazesPath);
			BufferedReader br = new BufferedReader(fr);
			String line = null;
			while((line = br.readLine()) != null){
				String coors[] = line.split(",");
				int x = Integer.parseInt(coors[0]);
				int y = Integer.parseInt(coors[1]);
				gazes.add(new Point(x,y));
			}
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("getGaze error");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.out.println("getGaze error");
		}
		return gazes;

	}
	
	protected String[] BagMILPath2ETGazesPath(BagMIL x, Integer h){
		String featurePath[] = x.getFileFeature(h).split("/");
		String ETLossFileName = featurePath[featurePath.length - 1];
		String imageFileName[] = x.getName().split("/");
		String imClass = imageFileName[imageFileName.length - 1].split("_")[0];
		String root = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
		//String ETBBPath = root + "gt_bb/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 
		// Where we stock gazes
		String ETLossPath =  root + "ETLoss_ratio/"+ imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
		String ETGazesPath = root + "gazes/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ;
		return new String[]{ETGazesPath, ETLossPath};
	}
	
	protected double getETLossRatio_help(String ETGazesPath,Rectangle PBB){
		ArrayList<Point> gazes = getGazes(ETGazesPath);
		double gazeNumber = 0;
		double inGazeNumber = 0;
		for(Point p: gazes){
			if (PBB.contains(p)){
				inGazeNumber+=1.0;
			}
			gazeNumber+=1.0;
		}
		return 1-inGazeNumber/gazeNumber;
	}
	
	protected double getETLossRatio(BagMIL x, Integer h, String ETGazesPath) {
		try {
			Iterator<ImageReader> readers = ImageIO  
                .getImageReadersByFormatName("JPG");  
        ImageReader reader = readers.next();  
        File bigFile = new File(x.getName());  
        ImageInputStream iis = ImageIO.createImageInputStream(bigFile);  
        reader.setInput(iis, true);  
		int	width = reader.getWidth(0);
		int height = reader.getHeight(0);
		int totalPatchNum = x.getFeatures().size();
		iis.close();
		int[] patch = getPatch(totalPatchNum, h);
		//ArrayList<Rectangle> ETBB = getETBB(ETGazesPath);
		Rectangle PBB = getPatchBB(width, height, totalPatchNum, patch);
		//calculate total surface as denominator
		return getETLossRatio_help(ETGazesPath, PBB);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println(x.getName() + "is bad");
			e.printStackTrace();
			return -10000.0;
		}  
	}
	

	//////////////////////////

	
	public double testAP(List<STrainingSample<LatentRepresentation<BagMIL, Integer>, Integer>> l, int scale, String simDir, String className) {
             
		List<Evaluation<Integer>> eval = new ArrayList<Evaluation<Integer>>();
        for(int i=0; i<l.size(); i++) {
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
