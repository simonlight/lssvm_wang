package latent.mantra.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import latent.LatentRepresentation;
import latent.LatentStructuralClassifier;
import struct.STrainingSample;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

public abstract class MantraMulticlass<X,H> implements LatentStructuralClassifier<X,Integer,H>{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2056862337482736692L;
	
	protected int optim = 1;
	protected double lambda = 1e-4;
	protected int cpmax = 50;
	protected int cpmin = 5;
	protected double epsilon = 1e-2;
	
	//svm hyperplane
	protected double[][] w = null;
	
	protected List<Integer> listClass = null;
	
	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();
	
	protected abstract List<H> enumerateH(X x);
	protected abstract double[] psi(X x, H h);
	protected abstract double delta(int y, int yp);
	protected abstract H[] init(STrainingSample<LatentRepresentation<X,H>,Integer> ts);
	
	@Override
	public Integer prediction(LatentRepresentation<X,H> rep) {
		return (int)valueOf(rep);
	}

	@Override
	public void train(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		
		if(l.isEmpty())
			return;
		
		int nbClass = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;
		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}
		double[] nb = new double[nbClass];
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
			nb[ts.output]++;
		}
		
		int dim = psi(l.get(0).input.x,l.get(0).input.h).length;
		
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train MANTRA multiclass - Mosek \tlambda= " + lambda + "\tdim= " + (nbClass*dim) + "\tnb class= " + nbClass + "\tdim= " + psi(l.get(0).input.x,l.get(0).input.h).length);
		System.out.println("class: " + listClass + "\t" + Arrays.toString(nb));
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println(optim + " - optim non convex \t Iterative - Cutting-Plane 1 Slack - primal-dual");
		}
		else if(optim == 2) {
			System.out.println(optim + " - optim convex \t Iterative - Cutting-Plane 1 Slack - primal-dual");
		}
		System.out.println("----------------------------------------------------------------------------------------");
	
		w = new double[nbClass][dim];
		
		long startTime = System.currentTimeMillis();
		if(optim == 1) {
			trainIterNonConvexCP1SlackPrimalDualMax(l);
		}
		else if(optim == 2) {
			trainIterConvexCP1SlackPrimalDualMax(l);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
		System.out.println("final obj= " + primalObj(l));
		accuracy(l);
			
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim latent - Time learning= "+ (endTime-startTime)/1000 + "s");
		
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	protected void trainIterNonConvexCP1SlackPrimalDualMax(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		
		double c = 1/lambda;
		int t=0;
		
		List<double[][]> la = new ArrayList<double[][]>();
		List<Double> lb = new ArrayList<Double>();
		List<ComputedScoresMinMax<H>> lr = null;
		List<double[][]> lw = new ArrayList<double[][]>();
		List<Double> lv = new ArrayList<Double>();
	
		lr = computeMinMaxInit(l, w);
		Object[] or = cuttingPlane(l, lr);
		double[][] at = (double[][]) or[0];
		double bt = (Double) or[1];
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && dot(w,at) < bt - xi - epsilon)) {
			
			System.out.print(".");
			if(t == cpmax) {
				System.out.print(" # max iter ");
			}
			
			if(t>0) {
				double[][] wstar = argminW(lw, lv);
				or = solveConflict( l, at, bt, w, wstar);
				at = (double[][]) or[0];
				bt = (Double) or[1];
			}
			la.add(at);
			lb.add(bt);
			
			if(gram != null) {
				double[][] g = gram;
				gram = new double[lb.size()][lb.size()];
				for(int i=0; i<g.length; i++) {
					for(int j=0; j<g.length; j++) {
						gram[i][j] = g[i][j];
					}
				}
				for(int i=0; i<lb.size(); i++) {
					gram[lb.size()-1][i] = dot(la.get(lb.size()-1), la.get(i));
					gram[i][lb.size()-1] = gram[lb.size()-1][i];
				}
				gram[lb.size()-1][lb.size()-1] += 1e-8;
			}
			else {
				gram = new double[lb.size()][lb.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = dot(la.get(j), la.get(i));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			double[] alphas = optimMosek(gram, la, lb, c);
			
			//System.out.println("DualObj= " + (dot(alphas,lb.toArray(new Double[lb.size()])) - 0.5 * matrixProduct(alphas,gram)) + "\talphas " + Arrays.toString(alphas));
			xi = (dot(alphas,lb.toArray(new Double[lb.size()])) - matrixProduct(alphas,gram))/c;
			
			// wtstar
			w = new double[la.get(0).length][la.get(0)[0].length];
			for(int i=0; i<alphas.length; i++) {
				for(int k=0; k<w.length; k++) {
					for(int d=0; d<w[k].length; d++) {
						w[k][d] += alphas[i] * la.get(i)[k][d];
					}
				}
			}
			lw.add(w);
			lv.add(primalObjCuttingPlane(la,lb,w));
			t++;
	
			computeMinMax(l, lr, w);
			or = cuttingPlane(l, lr);
			
			at = (double[][]) or[0];
			bt = (Double) or[1];
		}
		System.out.println("*");
	}
	
	public double[][] argminW(List<double[][]> lw, List<Double> lv) {
		double valMin = Double.MAX_VALUE;
		double[][] wstar = null;
		for(int i=0; i<lv.size(); i++) {
			double val = lv.get(i);
			if(val < valMin) {
				valMin = val;
				wstar = lw.get(i);
			}
		}
		return wstar;
	}
	
	public Object[] solveConflict(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, double[][] at, double bt, double[][] w, double[][] wstar) {
		
		double v1 = -dot(at,wstar) + bt;
		double v2 = loss(l, wstar);
		
		if(v1 > v2) {
			System.out.print("@");
			// compute U
			double U = loss(l, wstar) + dot(at,wstar);
			double obj = primalObj(l,wstar);
			double L = obj - lambda * dot(w,w) + dot(at,w);
			//System.out.println("v1= " + v1 + "\tv2= " + v2 + "\tL= " + L + "\tU= " + U + "\tb= " + bt);
			if(L <= U) {
				bt = L;
			}
			else {
				bt = L;
				for(int k=0; k<w.length; k++) {
					for(int d=0; d<w[k].length; d++) {
						at[k][d] = lambda*wstar[k][d];
					}
				}
			}
		}
		Object[] res = new Object[2];
		res[0] = at;
		res[1] = bt;
		return res;
	}
	
	protected double primalObjCuttingPlane(List<double[][]> la, List<Double> lb, double[][] w) {
		double obj = 0;
		for(int k=0; k<w.length; k++) {
			obj += linear.valueOf(w[k],w[k]);
		}
		obj *= lambda/2;
		double loss = -Double.MAX_VALUE;
		for(int i=0; i<la.size(); i++) {
			double val = -dot(la.get(i),w) + lb.get(i);
			if(val > loss) {
				loss = val;
			}
		}
		obj += loss;
		return obj;
	}
	
	protected void trainIterConvexCP1SlackPrimalDualMax(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		
		double c = 1/lambda;
		int t=0;
		
		List<double[][]> lg = new ArrayList<double[][]>();
		List<Double> lc = new ArrayList<Double>();
		List<ComputedScoresMinMax<H>> lr = null;
		
		lr = computeMinMaxInit(l, w);
		Object[] or = cuttingPlane(l, lr);
		double[][] gt = (double[][]) or[0];
		double ct = (Double) or[1];
		
		lg.add(gt);
		lc.add(ct);
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && dot(w,gt) < ct - xi - epsilon)) {
			
			System.out.print(".");
			if(t == cpmax) {
				System.out.print("# max iter");
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
						gram[i][j] = dot(lg.get(j), lg.get(i));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			
			
			double[] alphas = optimMosek(gram, lg, lc, c);
			
			//System.out.println("DualObj= " + (dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram)) + "\talphas " + Arrays.toString(alphas));
			
			//xi = (dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram))/c;
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
			
			computeMinMax(l, lr, w);
			or = cuttingPlane(l, lr);
			gt = (double[][]) or[0];
			ct = (Double) or[1];
			
			lg.add(gt);
			lc.add(ct);
		}
		
		System.out.println();
	}
	
	public Object[] cuttingPlane(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, List<ComputedScoresMinMax<H>> lr) {
		// compute g(t) and c(t)
		double[][] gt = new double[w.length][w[0].length];
		double ct = 0;
		double n = l.size();
		
		for(int i=0; i<l.size(); i++){
			STrainingSample<LatentRepresentation<X,H>,Integer> ts = l.get(i);			
			
			Object[] or = lossAugmentedInference(ts, lr.get(i));
			double[][] at = (double[][]) or[2];
			
			for(int y=0; y<w.length; y++) {
				for(int d=0; d<w[y].length; d++) {
					gt[y][d] += -at[y][d];
				}
			}
			ct += (Double) or[3];
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
	
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X,H>,Integer> ts, ComputedScoresMinMax<H> s) {
		
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		
		for(int y : listClass) {
			Object[] or = loss(ts,y,s,false);
			double val = (Double)or[0];
			//System.out.println(val);
			if(val>valmax){
				valmax = val;
				ypredict = y;
			}
		}
		
		Object[] or = loss(ts,ypredict,s,true);
		double ct = (Double)or[1];
		double[][] at = (double[][])or[2];
		
		Object[] res = new Object[4];
		res[0] = ypredict;
		res[1] = valmax;
		res[2] = at;
		res[3] = ct;
		return res;
	}
	
	
	protected Object[] loss(STrainingSample<LatentRepresentation<X,H>,Integer> ts, int y, ComputedScoresMinMax<H> s, boolean feature) {
		
		int ysec = s.getMaxY(y);
		int yprim = s.getMaxY(ts.output);
		
		double val = delta(ts.output, y) + s.getVmax(y) - s.getVmin(ysec) - s.getVmax(ts.output) + s.getVmin(yprim);
		double c = delta(ts.output, y);
		
		double[][] a = null;
		if(feature) {
			a = new double[w.length][w[0].length];
			if(val>0) {
				double[] psi1 = psi(ts.input.x,s.getHmax(y));
				double[] psi2 = psi(ts.input.x,s.getHmin(ysec));
				double[] psi3 = psi(ts.input.x,s.getHmax(ts.output));
				double[] psi4 = psi(ts.input.x,s.getHmin(yprim));
				for(int d=0; d<w[0].length; d++) {
					a[y][d] 		+= psi1[d];
					a[ysec][d] 		-= psi2[d];
					a[ts.output][d] 	-= psi3[d];
					a[yprim][d] 	+= psi4[d];
				}
			}
		}
		
		Object[] or = new Object[3];
		or[0] = val; //Math.max(0, val);
		or[1] = c;
		or[2] = a;
		return or;
	}
	
	protected List<ComputedScoresMinMax<H>> computeMinMaxInit(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, double[][] w) {
		List<ComputedScoresMinMax<H>> ls = new ArrayList<ComputedScoresMinMax<H>>();
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l){
			ComputedScoresMinMax<H> s = new ComputedScoresMinMax<H>();
			for(Integer y :listClass) {
				H[] hinit = init(ts);
				s.add(hinit[0], 0, hinit[1], 0);
			}
			ls.add(s);
		}
		return ls;
	}
	
	@SuppressWarnings("unchecked")
	protected void computeMinMax(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, List<ComputedScoresMinMax<H>> ls, double[][] w) {
		for(int i=0; i<l.size(); i++){
			STrainingSample<LatentRepresentation<X,H>,Integer> ts = l.get(i);
			for(Integer y :listClass) {
				Object[] or = valueOfHPlusMoins(ts.input, y, w);
				ls.get(i).set(y, (H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
			}
		}
	}
	
	public double valueOf(LatentRepresentation<X,H> rep) {
		
		double scoreMax = -Double.MAX_VALUE;
		double ypredict = -1;
		
		ComputedScoresMinMax<H> s = new ComputedScoresMinMax<H>();
		for(Integer y :listClass) {
			Object[] or = valueOfHPlusMoins(rep, y, w);
			s.add((H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
		}
		
		for(Integer y :listClass) {
			double score = valueOf(rep, y, s);
			if(score > scoreMax) {
				scoreMax = score;
				ypredict = y;
			}
		}
		
		return ypredict;
	}
	
	public double valueOf(LatentRepresentation<X,H> rep, int y, ComputedScoresMinMax<H> s) {
		int yprim = s.getMaxY(y);
		double score = s.getVmax(y) - s.getVmin(yprim);
		return score;
	}
	
	public double valueOf(LatentRepresentation<X,H> rep, int y) {
		ComputedScoresMinMax<H> s = new ComputedScoresMinMax<H>();
		for(Integer yprim :listClass) {
			Object[] or = valueOfHPlusMoins(rep, yprim, w);
			s.add((H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
		}
		double score = valueOf(rep, y, s);
		return score;
	}
	
	public Object[] valueOfH(LatentRepresentation<X,H> rep, double[][] w) {
		double scoreMax = -Double.MAX_VALUE;
		H hpredict = null;
		
		ComputedScoresMinMax<H> s = new ComputedScoresMinMax<H>();
		for(Integer y :listClass) {
			Object[] or = valueOfHPlusMoins(rep, y, w);
			s.add((H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
		}
		
		for(Integer y :listClass) {
			int yprim = s.getMaxY(y);
			double score = s.getVmax(y) - s.getVmin(yprim);
			if(score > scoreMax) {
				scoreMax = score;
				hpredict = s.getHmax(y);
			}
		}
		
		Object[] res = new Object[2];
		res[0] = scoreMax;
		res[1] = hpredict;
		return res;
	}
	
	protected Object[] valueOfHPlusMoins(LatentRepresentation<X,H> t, int y, double[][] w) {
		H hmax = null;
		H hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(H h : enumerateH(t.x)) {
			double[] psi = psi(t.x,h);
			double val = linear.valueOf(psi,w[y]);
			if(val>valmax){
				valmax = val;
				hmax = h;
			}
			if(val<valmin){
				valmin = val;
				hmin = h;
			}
		}
		Object[] res = new Object[4];
		res[0] = hmax;
		res[1] = valmax;
		res[2] = hmin;
		res[3] = valmin;
		return res;
	}
	
	protected double loss(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, double[][] w) {
		double loss = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
			ComputedScoresMinMax<H> s = new ComputedScoresMinMax<H>();
			for(Integer yprim :listClass) {
				Object[] or = valueOfHPlusMoins(ts.input, yprim, w);
				s.add((H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
			}
			Object[] or = lossAugmentedInference(ts, s);
			loss += (Double) or[1];
		}
		loss /= l.size();
		return loss;	
	}
	
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		double obj = 0;
		for(int k=0; k<w.length; k++) {
			obj += linear.valueOf(w[k],w[k]);
		}
		obj *= lambda/2;
		double loss = loss(l, w);
		System.out.println("lambda/2*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}
	
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l, double w[][]) {
		double obj = 0;
		for(int k=0; k<w.length; k++) {
			obj += linear.valueOf(w[k],w[k]);
		}
		obj *= lambda/2;
		double loss = loss(l, w);
		//System.out.println("lambda/2*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}
	
	protected double accuracy(List<STrainingSample<LatentRepresentation<X, H>, Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l){
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
			
			task.set_Stream(mosek.Env.streamtype.log, new mosek.Stream() {@Override
			public void stream(String msg) {}});
		
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
}
