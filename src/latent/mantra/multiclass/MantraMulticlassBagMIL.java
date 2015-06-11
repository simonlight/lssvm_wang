package latent.mantra.multiclass;

import io.FileWriterTxt;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;
import fr.lip6.jkernelmachines.util.DebugPrinter;

public class MantraMulticlassBagMIL extends MantraMulticlass<BagMIL,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7644028208486903431L;
	
	protected int initType = 1;
	
	public MantraMulticlassBagMIL() {
		super();
		w = null;
		DebugPrinter.DEBUG_LEVEL=0;
	}

	@Override
	protected List<Integer> enumerateH(BagMIL x) {
		List<Integer> latent = new ArrayList<Integer>();
		for(int i=0; i<x.getFeatures().size(); i++) {
			latent.add(i);
		}
		return latent;
	}

	protected double[] psi(BagMIL x, Integer h) {
		return x.getFeature(h);
	}

	@Override
	protected Integer[] init(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts) {
		Integer[] hinit = new Integer[2];
		if(initType == 0) {
			hinit[0] = 0;
			hinit[1] = 0;//ts.sample.x.getFeatures().size()-1;
		}
		else if(initType == 1) {
			hinit[0] = (int)(Math.random()*ts.input.x.getFeatures().size());
			hinit[1] = (int)(Math.random()*ts.input.x.getFeatures().size());
		}
		else if(initType == 2) {
			hinit[0] = 0;
			hinit[1] = (int)(Math.random()*ts.input.x.getFeatures().size());
		}
		else {
			System.out.println("error init");
			System.exit(0);
		}
		return hinit;
	}
	
	public double test(List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> l) {
		double[] nb = new double[listClass.size()];
		for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : l) {
			nb[ts.output]++;
		}
		System.out.println("Test - class: " + listClass + "\t" + Arrays.toString(nb));
		return accuracy(l);
	}
	
	public double testAP(List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> l) {
		double[] scores = new double[l.size()];
		for(int i=0; i<l.size(); i++) {
			int ypredict = (int) valueOf(l.get(i).input);
			double score = valueOf(l.get(i).input, ypredict);
			scores[i] = (ypredict==0 ? -1 : 1) * score;
		}
		
		List<Evaluation<Integer>> eval = new ArrayList<Evaluation<Integer>>();
		for(int i=0; i<scores.length; i++) {
			eval.add(new Evaluation<Integer>((l.get(i).output == 0 ? -1 : 1), scores[i]));
			//System.out.println(exampleTest.get(i).label + "\t" + scores[i] + ";");
		}
		double ap = AveragePrecision.getAP(eval);
		return ap;
	}

	@Override
	protected double delta(int y, int yp) {
		if(y == yp) {
			return 0;
		}
		else {
			return 1;
		}
	}
	
	public void save(File file) {
		
		System.out.println("save classifier: " + file.getAbsoluteFile());
		file.getParentFile().mkdirs();
		
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
		
			for(int i=0; i<w.length; i++) {
				bw.write("\n");
				for(double d : w[i]){
					bw.write(d + "\t");
		        }
			}
			bw.write("\nlambda\n" + lambda);
			bw.write("\ninit\n" + initType);
			bw.write("\noptim\n" + optim);
			bw.write("\nepsilon\n" + epsilon);
			bw.write("\ncpmax\n" + cpmax);
			bw.write("\ncpmin\n" + cpmin);
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
	
	public void load(File file) {
		
		System.out.println("load classifier: " + file.getAbsoluteFile());
		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);
			
			String ligne;
			ligne=br.readLine(); //"w"
			
			List<List<Double>> list = new ArrayList<List<Double>>();
			int n=0;
			while((ligne=br.readLine()) != null && ligne.compareToIgnoreCase("lambda") != 0) {
				StringTokenizer st = new StringTokenizer(ligne);
				list.add(new ArrayList<Double>());
				while(st.hasMoreTokens()) {
					list.get(n).add(Double.parseDouble(st.nextToken()));
				}
				n++;
			}
			w = new double[list.size()][list.get(0).size()];
			for(int i=0; i<list.size(); i++) {
				for(int j=0; j<list.get(i).size(); j++) {
					w[i][j] = list.get(i).get(j);
				}
			}
			System.out.println("w " + w.length + " x " + w[0].length);
			
			listClass = new ArrayList<Integer>();
			for(int i=0; i<w.length; i++) {
				listClass.add(i);
			}
			
			//ligne=br.readLine(); //"lambda"
			ligne=br.readLine();
			lambda = Double.parseDouble(ligne);
			
			ligne=br.readLine(); //"initType"
			ligne=br.readLine();
			initType = Integer.parseInt(ligne); 
			
			ligne=br.readLine(); //"optim"
			ligne=br.readLine();
			optim = Integer.parseInt(ligne); 
			
			ligne=br.readLine(); //"epsilon"
			ligne=br.readLine();
			epsilon = Double.parseDouble(ligne);
			
			ligne=br.readLine(); //"cpmax"
			ligne=br.readLine();
			cpmax = Integer.parseInt(ligne); 
			
			ligne=br.readLine(); //"cpmin"
			ligne=br.readLine();
			cpmin = Integer.parseInt(ligne); 
			
			br.close();
		}
		catch (IOException e) {
			System.out.println(e);
			System.out.println("Error parsing file " + file);
		}
		
		showParameters();
	}
	
	public List<Double> getScoresAllClass(LatentRepresentation<BagMIL, Integer> sample) {
		List<Double> scores = new ArrayList<Double>();
		for(int y : listClass) {
			double score = valueOf(sample,y);
			scores.add(score);
		}
		
		return scores;
	}

	
	public void writeResults(File file, List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write results " + file.getAbsolutePath());
		
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
			
			bw.write("name \t gt \t ypredict \t hpredict \t score \t fileFeature \n");
			for(int i=0; i<l.size(); i++){
        		bw.write(l.get(i).input.x.getName() + "\t" + l.get(i).output + "\t");
        		int ypredict = (int) valueOf(l.get(i).input);
        		Object[] or =  valueOfH(l.get(i).input, w);
        		double score = (Double) or[0];
    			int hpredict = (Integer) or[1];
    			bw.write(ypredict + "\t" + hpredict + "\t" + score + "\t");
        		bw.write(l.get(i).input.x.getFileFeature(hpredict) + "\n");
	        }
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
	
	public void writeScores(File file, List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write scores " + file.getAbsolutePath());
		
		if(!file.exists()) {
			List<List<Double>> listScore = new ArrayList<List<Double>>();
			for(int n=0; n<l.size(); n++) {
				List<Double> sc = getScoresAllClass(l.get(n).input);
				listScore.add(n,sc);
			}
			
			FileWriterTxt.writeSignature(listScore, file);
			System.err.println("write file " + file.getAbsolutePath());
		}
	}
	
	public int getInitType() {
		return initType;
	}

	public void setInitType(int initType) {
		this.initType = initType;
	}
	
	public void showParameters(){
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train MANTRA multiclass - Mosek \tlambda= " + lambda + "\tdim= " + w.length*w[0].length + "\tnb class= " + w.length + "\tdim= " + w[0].length);
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println(optim + " - optim convex \t Iterative Max - Cutting-Plane 1 Slack - primal-dual");
		}
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	public String toString() {
		return "mantra_multiclass_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin + "_init_" + initType;
	}
}
