package ssvm.multiclass;

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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

import struct.STrainingSample;
import struct.StructuralClassifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

public class SSVMMulticlassFast implements StructuralClassifier<double[], Integer>, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -712338181556012795L;
	
	protected int optim = 1;
	protected double lambda = 1e-4;
	protected int cpmax = 50;
	protected int cpmin = 5;
	protected double epsilon = 1e-2;
	
	// debug
	DebugPrinter debug = new DebugPrinter();
	
	//svm hyperplane
	protected double[][] w = null;
	protected List<Integer> listClass = null;
	
	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();
	
	protected double delta(int y, int yp) {
		if(y == yp) {
			return 0;
		}
		else {
			return 1;
		}
	}

	@Override
	public Integer prediction(double[] x) {
		return (int) valueOf(x);
	}

	@Override
	public void train(List<STrainingSample<double[], Integer>> l) {
		
		if(l.isEmpty())
			return;
		
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
		
		int dim = l.get(0).input.length;
		
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train SSVM Multiclass \tlambda: " + lambda + "\tfinal dim: " + (nbClass*dim) + "\tnb class: " + nbClass + "\tdim: " + dim);
		System.out.println("class: " + listClass + "\t" + Arrays.toString(nb));
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println("optim " + optim + " - Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("----------------------------------------------------------------------------------------");
		
		w = new double[nbClass][dim];
		
		long startTime = System.currentTimeMillis();
		if(optim == 1) {
			trainCP1SlackPrimalDual(l);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
		System.out.println("obj= " + primalObj(l,w));
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		accuracy(l);
		
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	protected void trainCP1SlackPrimalDual(List<STrainingSample<double[], Integer>> l) {
		
		double c = 1/lambda;
		int t=0;
		
		List<double[][]> lg = new ArrayList<double[][]>();
		List<Double> lc 	= new ArrayList<Double>();

		Object[] or 	= cuttingPlane(l,w);
		double[][] gt 	= (double[][]) or[0];
		double ct		= (Double) or[1];
		
		lg.add(gt);
		lc.add(ct);
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && dot(w,gt) < ct - xi - epsilon)) {
			
			System.out.print(".");
			if(t == cpmax) {
				System.out.print(" # max iter ");
			}
			
			if(gram != null) {
				double[][] g = gram;
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<g.length; i++) {
					for(int j=0; j<g.length; j++) {
						gram[i][j] = g[i][j];
					}
				}
				for(int i=0; i<lc.size(); i++) {
					gram[lc.size()-1][i] = dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = dot(lg.get(i), lg.get(j));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			double[] alphas = optimMosek(gram, lg, lc, c);
			xi = (dot(alphas,lc.toArray(new Double[lc.size()])) - matrixProduct(alphas,gram))/c;

			// new w
			w = new double[lg.get(0).length][lg.get(0)[0].length];
			for(int i=0; i<alphas.length; i++) {
				for(int k=0; k<gt.length; k++) {
					for(int d=0; d<gt[k].length; d++) {
						w[k][d] += alphas[i] * lg.get(i)[k][d];
					}
				}
			}
			t++;

			or = cuttingPlane(l, w);
			gt = (double[][]) or[0];
			ct = (Double) or[1];
			
			lg.add(gt);
			lc.add(ct);
			
		}
		System.out.println("*");
	}
	
	public Object[] cuttingPlane(List<STrainingSample<double[], Integer>> l, double[][] w) {
		// compute g(t) and c(t)
		double[][] gt = new double[w.length][w[0].length];
		double ct = 0;
		double n = l.size();
		
		for(STrainingSample<double[], Integer> ts : l){		
			Object[] or = lossAugmentedInference(ts, w);
			ct += (Double) or[3];
			double[][] at = (double[][]) or[2];
			
			for(int y=0; y<w.length; y++) {
				for(int d=0; d<w[y].length; d++) {
					gt[y][d] += -at[y][d];
				}
			}
		}
		ct /= n;
		
		for(int k=0; k<gt.length; k++) {
			for(int d=0; d<gt[k].length; d++) {
				gt[k][d] /= n;
			}
		}
		
		Object[] res = new Object[2];
		res[0] = gt;
		res[1] = ct;
		return res;
	}
	
	protected Object[] lossAugmentedInference(STrainingSample<double[], Integer> ts, double[][] w) {
		
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			Object[] or = loss(ts, y, false, w);
			double val = (Double)or[0];
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		
		Object[] or = loss(ts, ypredict, true, w);
		double bt = (Double)or[1];
		double[][] at = (double[][])or[2];
		
		Object[] res = new Object[4];
		res[0] = ypredict;
		res[1] = valmax;
		res[2] = at;
		res[3] = bt;
		return res;
	}
	
	protected Object[] loss(STrainingSample<double[], Integer> ts, int y, boolean feature) {
		return loss(ts, y, feature, w);
	}
	
	protected Object[] loss(STrainingSample<double[], Integer> ts, int y, boolean feature, double[][] w) {
		
		double val = delta(ts.output, y) + linear.valueOf(ts.input, w[y]) - linear.valueOf(ts.input, w[ts.output]);
		double b = delta(ts.output, y);
		
		double[][] a = null;
		if(feature) {
			a = new double[w.length][w[0].length];
			double[] psi = ts.input;
			for(int d=0; d<w[y].length; d++) {
				a[y][d] 		+= psi[d];
				a[ts.output][d] 	-= psi[d];
			}
		}
		
		Object[] or = new Object[3];
		or[0] = val;
		or[1] = b;
		or[2] = a;
		return or;
	}
	
	public double valueOf(double[] rep) {
		double scoreMax = -Double.MAX_VALUE;
		double ypredict = -1;
		for(int y : listClass) {
			double score = valueOf(rep, y);
			if(score > scoreMax) {
				scoreMax = score;
				ypredict = y;
			}
		}
		return ypredict;
	}
	
	public double valueOf(double[] rep, int y) {
		return valueOf(rep, y, w);
	}
	
	public double valueOf(double[] rep, int y, double[][] w) {
		double score = linear.valueOf(rep, w[y]);
		return score;
	}
	
	protected double loss(List<STrainingSample<double[], Integer>> l, double[][] w) {
		double loss = 0;
		for(STrainingSample<double[], Integer> ts : l) {
			Object[] or = lossAugmentedInference(ts, w);
			loss += (Double) or[1];
		}
		loss /= l.size();
		return loss;	
	}
	
	protected double primalObj(List<STrainingSample<double[], Integer>> l, double[][] w) {
		double obj = lambda * dot(w,w)/2;
		double loss = loss(l,w);
		System.out.println("lambda/2*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}
	
	public double primalObj(List<STrainingSample<double[], Integer>> l) {
		return primalObj(l, w);
	}
	
	public double test(List<STrainingSample<double[], Integer>> l) {
		return accuracy(l);
	}
	
	protected double accuracy(List<STrainingSample<double[], Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<double[], Integer> ts : l){
			int ypredict = (int) valueOf(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}
	
	protected double[] optimMosek(double[][] gram, List<double[][]> lg, List<Double> lc, double c) {
		
		mosek.Env env = null;
		mosek.Task task = null;
		
		double[] alphas = null;
		
		try {
			env = new mosek.Env();
			task = new mosek.Task(env,0,0);
			
			task.set_Stream(mosek.Env.streamtype.log, new mosek.Stream() {public void stream(String msg) {}});
		
			int numcon = 1;
			task.appendcons(numcon); // number of constraints
			
			int numvar = lc.size();	// number of variables
			task.appendvars(numvar);	
			
			for(int i=0; i<numvar; i++) {
				// linear term in the objective
				task.putcj(i, -lc.get(i));
				
				// bounds on variable i
				task.putbound(mosek.Env.accmode.var, i, mosek.Env.boundkey.ra, 0.0, c);
				
				//
				int[] asub = {0};
				double[] aval = {1.};
				task.putacol(i,asub,aval);
			}
			
			// bounds on constraints 
			for(int i=0; i<numcon; i++) {	
				task.putbound(mosek.Env.accmode.con, i, mosek.Env.boundkey.ra, 0., c);
			}
		
			//The lower triangular part of the Q matrix in the objective is specified. 
			int[] qi = new int[numvar*(numvar+1)/2];
			int[] qj = new int[numvar*(numvar+1)/2];
			double[] qval = new double[numvar*(numvar+1)/2];
			
			int n=0;
			for(int i=0; i<lc.size(); i++) {
				for(int j=0; j<=i; j++) {
					qi[n] = i;
					qj[n] = j;
					qval[n] = gram[i][j];
					n++;
				}
			}
			
			// Input the Q for the objective
			task.putqobj(qi, qj, qval);
			
			// Solve the problem
		    mosek.Env.rescode r = task.optimize(); 
		    //System.out.println(" Mosek warning:" + r.toString());
		    
		    // Print a summary containing information about the solution for debugging purposes 
		    task.solutionsummary(mosek.Env.streamtype.msg); 
		    
		    mosek.Env.solsta solsta[] = new mosek.Env.solsta[1]; 
		    // Get status information about the solution 
		    task.getsolsta(mosek.Env.soltype.itr,solsta); 
		    
		    // Get the solution
		    alphas = new double[numvar];
		    task.getxx(mosek.Env.soltype.itr, alphas); 
		    
		    //System.out.println("MOSEK - alphas " + Arrays.toString(alphas));
		}
	    catch (mosek.Exception e) { 
	      System.out.println ("An error/warning was encountered"); 
	      System.out.println (e.toString()); 
	      throw e; 
	    } 
	    finally { 
		    if(task != null) {
		    	task.dispose(); 
		    }
		    if(env != null) {
		    	env.dispose();
		    }
	    }
		
		return alphas;
	}
	
	protected double dot(double[][] a, double[][] b) {
		double s = 0;
		for(int i=0; i<a.length; i++) {
			s += VectorOperations.dot(a[i], b[i]);
		}
		return s;
	}
	
	protected double dot(double[] a, Double[] b) {
		double s = 0;
		for(int i=0; i<a.length; i++) {
			s += a[i] * b[i];
		}
		return s;
	}
	
	protected double matrixProduct(double[] alphas, double[][] gram) {
		// alpha^T*Gramm*alpha
		double[] tmp = new double[alphas.length];
		// tmp = gram * alpha
		for(int i=0; i<gram.length; i++) {
			tmp[i] = VectorOperations.dot(gram[i],alphas);
		}
		double s = VectorOperations.dot(alphas,tmp);
		return s;
	}
	
	protected double[][] add(double[][] a, double[][] b) {
		// a = a + b
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				a[i][j] += b[i][j];
			}
		}
		return a;
	}
	
	protected double[][] add(double[][] a, double[][] b, double c) {
		// a = a + b*c
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				a[i][j] += b[i][j] * c;
			}
		}
		return a;
	}
	
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	public double[][] getW() {
		return w;
	}
	public void setW(double[][] w) {
		this.w = w;
	}
	public int getOptim() {
		return optim;
	}
	public void setOptim(int optim) {
		this.optim = optim;
	}
	public int getCpmax() {
		return cpmax;
	}
	public void setCpmax(int cpmax) {
		this.cpmax = cpmax;
	}
	public int getCpmin() {
		return cpmin;
	}
	public void setCpmin(int cpmin) {
		this.cpmin = cpmin;
	}
	public double getEpsilon() {
		return epsilon;
	}
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}
	
	public String toString() {
		return "ssvm_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin;
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
	
	public void showParameters(){
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train SSVM Multiclass \tlambda: " + lambda + "\tdim: " + w.length*w[0].length + "\tnb class: " + w.length + "\tdim: " + w[0].length);
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println("optim " + optim + " - Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("----------------------------------------------------------------------------------------");
	}
}
