package ssvm.ranking;

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
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

import ssvm.ranking.nico.Pair;
import struct.STrainingSample;
import struct.StructuralClassifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

public class SSVMRankAP implements StructuralClassifier<RankFeature, List<Integer>>, Serializable {
	
	/**
	 * Learn to rank documents where the binary labels are relevant(+1) and non-relevant(-1)
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
	protected double[] w = null;
	protected List<Integer> listClass = null;
	
	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();
	
	protected double[] psigt = null;

	@Override
	public void train(List<STrainingSample<RankFeature, List<Integer>>> l) {
		if(l.isEmpty())
			return;
		
		train(l.get(0));
	}
	
	public void train(STrainingSample<RankFeature, List<Integer>> ts) {
		
		w = new double[ts.input.getDim()];
		
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train SSVM Ranking \tlambda: " + lambda + "\tdim: " + w.length);
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println("optim " + optim + " - Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("----------------------------------------------------------------------------------------");
		
		w[0] = 1;
		System.out.println("init ap= " + test(ts));
		long startTime = System.currentTimeMillis();
		if(optim == 1) {
			trainCP1SlackPrimalDual(ts);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
		System.out.println("obj= " + primalObj(ts,w));
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("final ap= " + test(ts));
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	protected void trainCP1SlackPrimalDual(STrainingSample<RankFeature, List<Integer>> ts) {
		double c = 1/lambda;
		int t=0;
		
		psigt = psi(ts.input, ts.output);
		//System.out.println(Arrays.toString(psigt));
		
		List<double[]> lg 	= new ArrayList<double[]>();
		List<Double> lc 	= new ArrayList<Double>();

		Object[] or 	= cuttingPlane(ts,w);
		double[] gt 	= (double[]) or[0];
		double ct		= (Double) or[1];
		
		lg.add(gt);
		lc.add(ct);
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && VectorOperations.dot(w,gt) < ct - xi - epsilon)) {
			
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
					gram[lc.size()-1][i] = VectorOperations.dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = VectorOperations.dot(lg.get(i), lg.get(j));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			double[] alphas = optimMosek(gram, lg, lc, c);
			//System.out.println("alphas " + Arrays.toString(alphas));
			xi = (dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram))/c;
			
			// new w
			w = new double[lg.get(0).length];
			for(int i=0; i<alphas.length; i++) {
				for(int d=0; d<gt.length; d++) {
					w[d] += alphas[i] * lg.get(i)[d];
				}
			}
			t++;
			//System.out.println("w= " + Arrays.toString(w));
			//System.out.println("ap= " + test(ts));
			
			or = cuttingPlane(ts, w);
			gt = (double[]) or[0];
			ct = (Double) or[1];
			
			lg.add(gt);
			lc.add(ct);
		}
		System.out.println("*");
	}
	
	protected double[] psi(RankFeature x, List<Integer> y) {
		double[] psi = new double[x.getDim()];
		for(int i : x.getLp()) {
			for(int j : x.getLn()) {
				if(y.get(i)<y.get(j)) {
					for(int d=0; d<psi.length; d++) {
						psi[d] += x.getFeatures().get(i)[d] - x.getFeatures().get(j)[d];
					}
				}
				else {
					for(int d=0; d<psi.length; d++) {
						psi[d] += x.getFeatures().get(j)[d] - x.getFeatures().get(i)[d];
					}
				}
			}
		}
		double tmp = 1/(1.* x.getLp().size() * x.getLn().size());
		for(int d=0; d<psi.length; d++) {
			psi[d] *= tmp;
		}
		//System.out.println("y= " + y);
		//System.out.println("psi= " + Arrays.toString(psi));
		return psi;
	}
	
	public Object[] cuttingPlane(STrainingSample<RankFeature, List<Integer>> ts, double[] w) {
		// compute g(t) and c(t)
		double[] gt = new double[w.length];
		double ct = 0;

		Object[] or = lossAugmentedInference(ts, w);
		ct += (Double) or[3];
		double[] at = (double[]) or[2];
		
		for(int d=0; d<w.length; d++) {
			gt[d] += -at[d];
		}
		
		Object[] res = new Object[2];
		res[0] = gt;
		res[1] = ct;
		return res;
	}
	
	protected Object[] lossAugmentedInference(STrainingSample<RankFeature, List<Integer>> ts, double[] w) {
		
		// Sorting + in descending order of <w ; xi>
		List<Pair<Integer,Double>> sortedP = new ArrayList<Pair<Integer,Double>>();
		for(int i : ts.input.getLp()){
			sortedP.add(new Pair<Integer,Double>(i, linear.valueOf(w, ts.input.getFeatures().get(i))));
		}
		Collections.sort(sortedP,Collections.reverseOrder());
		
		// Sorting - in descending order of <w ; xi>
		List<Pair<Integer,Double>> sortedN = new ArrayList<Pair<Integer,Double>>();
		for(int i : ts.input.getLn()){
			sortedN.add(new Pair<Integer,Double>(i, linear.valueOf(w, ts.input.getFeatures().get(i))));
		}
		Collections.sort(sortedN,Collections.reverseOrder());
		
		List<Integer> indNeg = new ArrayList<Integer>();
		for(int j=1; j<=ts.input.getLn().size(); j++) {
			// pre-compute i-th delta value
			double[] deltaj = new double[ts.input.getLp().size()];
			for(int i=1; i<=ts.input.getLp().size(); i++){
				deltaj[i-1] = deltaJi(i, j, sortedP.get(i-1).getValue(), sortedN.get(j-1).getValue(), (double)ts.input.getLp().size(), (double)ts.input.getLn().size());
			}
			
			int imax = 0;
			double valmax = -Double.MAX_VALUE;
			double val = 0.0;
			for(int i=ts.input.getLp().size(); i>=1; i--){
				val += deltaj[i-1];
				if(val>=valmax){
					valmax = val;
					imax = i;
				}
			}
			indNeg.add(imax);
		}
		//System.out.println("ind neg " + indNeg);
		
		List<Integer> ystar = new ArrayList<Integer>();
		for(int i=0; i<ts.input.getLp().size(); i++){
			ystar.add(sortedP.get(i).getKey());
		}
		for(int i=ts.input.getLn().size()-1; i>=0; i--){
			ystar.add(indNeg.get(i),sortedN.get(i).getKey());
		}
		System.out.println("loss augmeted inference " + ystar);
		
		double bt = delta(ts.input.getGtLabels(), ystar);
		double[] at = psi(ts.input,ystar);
		System.out.println("ystar= " + (bt+linear.valueOf(w, at)) + "\tgt= " + linear.valueOf(w, psigt));
		for(int d=0; d<w.length; d++) {
			at[d] -= psigt[d];
		}
		//System.out.println("delta= " + bt);
		Object[] res = new Object[4];
		res[0] = ystar;
		res[1] = bt + linear.valueOf(w, at);
		res[2] = at;
		res[3] = bt;
		return res;
	}
	
	protected double deltaJi(int i, int j, double sip, double sjn, double p, double n) {
		return (double)((j/(j+i) - (j-1)/(j+i-1))/(1.*p) - 2.*(sip-sjn)/(1.*p*n));
	}
	
	public double delta(List<Integer> yi, List<Integer> y) {
		// Input Y : supposed to be list containing the ordered indices of examples (as stored in gtOrdering)
		// Yi label of all examples
		return 1.0 - ap(yi, y);
	}
	
	public double ap(List<Integer> yi, List<Integer> y /*, int nbPlus*/){
		
		int[] tp = new int[y.size()];
		int[] fp = new int[y.size()];
		
		int i = 0;
		int cumtp = 0, cumfp = 0;
		int totalpos = 0;
		
		//cumsum of true positives and false positives
		for(int j : y) {
			if(yi.get(j) == 1) {
				cumtp++;
				totalpos++;
			}
			else {
				cumfp++;
			}
			tp[i] = cumtp;
			fp[i] = cumfp;
			i++;
		}
		
		//precision / recall
		double[] prec = new double[tp.length];
		double[] reca = new double[tp.length];
		
		for(i = 0 ; i < tp.length ; i++) {
			reca[i] = ((double)tp[i])/((double)totalpos);
			prec[i] = ((double)tp[i])/((double)(tp[i]+fp[i]));
		}
		
		double[] mrec = new double[reca.length+2];
		for(int j=0; j<reca.length; j++) {
			mrec[j+1] = reca[j];
		}
		mrec[mrec.length-1] = 1;
		
		double[] mpre = new double[prec.length+2];
		for(int j=0; j<prec.length; j++) {
			mpre[j+1] = prec[j];
		}
		
		for(int j=mpre.length-2; j>=0; j--) {
		    mpre[j] = Math.max(mpre[j],mpre[j+1]);
		}
		
		double map = 0.;
		for(int j=1; j<mpre.length-1; j++) {
			map += (mrec[j]-mrec[j-1])*mpre[j];
		}
		
		return map;
	}
	
	public List<Integer> prediction(STrainingSample<RankFeature,List<Integer>> ts) {
		// y estimate = arg max_y {<w,psi(xi,y)>} 
		
		List<Integer> res = new ArrayList<Integer>();
		List<Pair<Integer,Double>> pairs = new ArrayList<Pair<Integer,Double>>();
		
		for(int i=0; i<ts.input.getFeatures().size(); i++){
			pairs.add(new Pair<Integer,Double>( i , VectorOperations.dot(w, ts.input.getFeatures().get(i))));
		}
		Collections.sort(pairs, Collections.reverseOrder());
		for(int i=0; i< ts.input.getFeatures().size(); i++){
			res.add(pairs.get(i).getKey());
		}

		System.out.println("inference " + res);
		return res;
	}
	
	public double test(List<STrainingSample<RankFeature,List<Integer>>> l) {
		List<Integer> y = prediction(l.get(0));
		return ap(l.get(0).input.getGtLabels(), y);
	}
	
	public double test(STrainingSample<RankFeature,List<Integer>> ts) {
		List<Integer> y = prediction(ts);
		return ap(ts.input.getGtLabels(), y);
	}
	
	protected double primalObj(STrainingSample<RankFeature, List<Integer>> l, double[] w) {
		double obj = lambda * VectorOperations.dot(w,w)/2;
		double loss = (Double)lossAugmentedInference(l,w)[1];
		System.out.println("lambda/2*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}
	
	public double primalObj(STrainingSample<RankFeature, List<Integer>> l) {
		return primalObj(l, w);
	}
	
	protected List<Integer> convertRankOrdering(List<Integer> list){
		// Input : ordered List (ex [2,0,1,3])
		// Output : rank of each element : 0=>1 1=>2 2=>0 3=>4 
		// This is symmetric, switiching input/output works
		List<Integer> res= new ArrayList<Integer>(Collections.nCopies(list.size(), 0));
		for(int i=0;i<res.size();i++){
			res.set(list.get(i), i);
		}
		return res;
	}
	
	protected double[] optimMosek(double[][] gram, List<double[]> lg, List<Double> lc, double c) {
		
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
	
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	public double[] getW() {
		return w;
	}
	public void setW(double[] w) {
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
		return "ssvm_rankAP_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin;
	}
	
	public void save(File file) {
		
		System.out.println("save classifier: " + file.getAbsoluteFile());
		file.getParentFile().mkdirs();
		
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
		
			bw.write("w\n");
			for(int i=0; i<w.length; i++) {
				bw.write(w[i] + "\t");
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
			w = new double[list.size()];
			for(int i=0; i<list.size(); i++) {
				for(int j=0; j<list.get(i).size(); j++) {
					w[j] = list.get(i).get(j);
				}
			}
			System.out.println("w " + w.length);
			
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
		System.out.println("Train SSVM Multiclass \tlambda: " + lambda + "\tdim: " + w.length );
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println("optim " + optim + " - Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("----------------------------------------------------------------------------------------");
	}
}
