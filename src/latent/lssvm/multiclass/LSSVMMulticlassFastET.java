package latent.lssvm.multiclass;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

import javax.imageio.ImageIO;

import latent.LatentRepresentation;
import latent.LatentStructuralClassifier;
import latent.variable.BagMIL;
import solver.MosekSolver;
import struct.STrainingSample;
import util.VectorOp;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

public abstract class LSSVMMulticlassFastET<X,H> implements LatentStructuralClassifier<X,Integer,H> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 987718067018611039L;
	
	protected int optim = 1;
	protected double lambda = 1e-4;
	protected int epochsLatentMax = 50;
	protected int epochsLatentMin = 2;
	protected int cpmax = 50;
	protected int cpmin = 5;
	protected double epsilon = 1e-2;

	//svm hyperplane
	protected double[][] w = null;
	protected int dim = 1;
	
	// list of the classes
	protected List<Integer> listClass = null;
	
	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();

	//eye_tracking params
	protected HashMap<String , Double> positiveLossMap = new HashMap<String , Double>(); 
	protected double tradeoff = 0.5;

	protected abstract List<H> enumerateH(X x);
	protected abstract double[] psi(X x, H h);
	protected abstract double delta(Integer yi, Integer yp, X x, H h);
	protected abstract double getETLoss(X x, H h);

	/**
	 * initialise la dimension de w (variable dim) et les variables latentes si nécessaire 
	 * @param l
	 */
	protected abstract void init(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l);

	@Override
	public void train(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		
		if(l.isEmpty())
			return;
	
		int nbClass = 0;
		for(STrainingSample<LatentRepresentation<X, H>,Integer> ts : l) {
			nbClass = Math.max(nbClass, ts.output);
		}
		nbClass++;
		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}
		double[] nb = new double[nbClass];
		for(STrainingSample<LatentRepresentation<X, H>,Integer> ts : l) {
			nb[ts.output]++;
		}
		
		System.out.println("Multiclass \t dim= " + dim + " \tclasses: " + listClass + "\t" + Arrays.toString(nb));
		
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train LSSVM - Mosek \tlambda: " + lambda + "\tepochsLatentMax " + epochsLatentMax + "\tepochsLatentMin " + epochsLatentMin); 
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		init(l);
		System.out.println("Multiclass \t dim= " + dim + " \tclasses: " + listClass + "\t" + Arrays.toString(nb));
		if(optim == 1) {
			System.out.println("optim " + optim + " \t CCCP - Cutting-Plane 1 Slack");
		}
		System.out.println("----------------------------------------------------------------------------------------");
		
		w = new double[listClass.size()][dim];
		
		long startTime = System.currentTimeMillis();
		if(optim == 1) {
			trainCCCP(l);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
		//System.out.println("obj= " + primalObj(l));
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim latent - Time learning= "+ (endTime-startTime)/1000 + "s");
		//System.out.println("Evaluation after training " + evaluation(l));
		
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	protected void trainCCCP(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		int el=0;
		double decrement = 0;
		double precObj = 0;
		while(el<epochsLatentMin || (el<=epochsLatentMax && decrement < 0)) {
			System.out.println("epoch latent " + el);
			trainCCCPCP1Slack(l);
			double obj = primalObj(l);
			decrement = obj - precObj;
			System.out.println("obj= " + obj + "\tdecrement= " + decrement);
			precObj = obj;
			el++;
			
			for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l){
				ts.input.h = prediction(ts.input.x,ts.output);
			}
		}
	}
	
	protected void trainCCCPCP1Slack(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		
		double c = 1/lambda;
		int t=0;
		
		List<double[][]> lg = new ArrayList<double[][]>();
		List<Double> lc = new ArrayList<Double>();
		
		Object[] or = cuttingPlane(l);
		double[][] gt = (double[][]) or[0];
		double ct = (Double) or[1];
		
		lg.add(gt);
		lc.add(ct);
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && VectorOp.dot(w,gt) < ct - xi - epsilon)) {
			
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
					gram[lc.size()-1][i] = VectorOp.dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = VectorOp.dot(lg.get(j), lg.get(i));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			// Solve the QP
			double[] alphas = MosekSolver.solveQP(gram, lc, c);
			//System.out.println("DualObj= " + (dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram)) + "\talphas " + Arrays.toString(alphas));
			xi = (VectorOp.dot(alphas,lc.toArray(new Double[lc.size()])) - matrixProduct(alphas,gram)) / c;
			
			// new w
			w = new double[listClass.size()][dim];
			for(int i=0; i<alphas.length; i++) {
				for(int k=0; k<w.length; k++) {
					for(int d=0; d<dim; d++) {
						w[k][d] += alphas[i] * lg.get(i)[k][d];
					}
				}
			}
			t++;

			or = cuttingPlane(l);
			gt = (double[][]) or[0];
			ct = (Double) or[1];
			
			lg.add(gt);
			lc.add(ct);
		}
		System.out.println(" Inner loop optimization finished.");
	}
	
	public Object[] cuttingPlane(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		// compute g(t) and c(t)
		double[][] gt = new double[w.length][w[0].length];
		double ct = 0;
		double n = l.size();
		for(int i=0; i<l.size(); i++){
			STrainingSample<LatentRepresentation<X,H>,Integer> ts = l.get(i);			
			Object[] or = lossAugmentedInference(ts);
			Integer yp = (Integer)or[0];
			H hp = (H)or[1];
			ct += delta(ts.output, yp, ts.input.x, hp);
			double[] psi1 = psi(ts.input.x, hp);
			double[] psi2 = psi(ts.input.x, ts.input.h);
			for(int d=0; d<w[ts.output].length; d++) {
				gt[yp][d] 			-= psi1[d];
				gt[ts.output][d] 	+= psi2[d];
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
	
	/**
	 * compute the loss of the objective function
	 * \sum_{i=1}^N max_{y,h} ( delta(yi,y) + <w, \psi(xi,y,h)> ) - max_hp <w,psi(xi,yi,hp)>
	 */
	protected double loss(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		double loss = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
			Object[] or = lossAugmentedInference(ts);
			Integer yp = (Integer)or[0];
			loss += delta(ts.output,yp, ts.input.x, (H)or[1]);
		}
		loss /= l.size();
		return loss;	
	}
	
	protected double primalObj(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
		double obj = lambda * VectorOp.dot(w,w)/2;
		double loss = loss(l);
		System.out.println("lambda*||w||^2= " + obj + "\t\tloss= " + loss);
		obj += loss;
		return obj;
	}
	
//	public double evaluation(List<STrainingSample<LatentRepresentation<X,H>,Integer>> l) {
//		double delta = 0;
//		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l) {
//			Integer yp = prediction(ts.input);
//			delta += delta(ts.output,yp);
//		}
//		return delta;
//	}
	
	public Integer prediction(LatentRepresentation<X, H> lr) {
		int ypredict = -1;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			for(H h : enumerateH(lr.x)) {
				
				double val = valueOf(lr.x,y,h,w)+tradeoff * (1-getETLoss(lr.x, h));
				if(val>valmax){
					valmax = val;
					ypredict = y;
				}
			}
		}
		return ypredict;
	}

	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X, H>, Integer> ts) {
		int ypredict = -1;
		H hpredict = null;
		double valmax = -Double.MAX_VALUE;
		for(int y : listClass) {
			for(H h : enumerateH(ts.input.x)) {
				double val = delta(ts.output, y, ts.input.x, h) + valueOf(ts.input.x,y,h,w);
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

	protected H prediction(X x, Integer y) {
		H hpredict = null;
		double valmax = -Double.MAX_VALUE;
		for(H h : enumerateH(x)) {
			double val = valueOf(x,y,h,w)+tradeoff * (1 - getETLoss(x, h));
			if(val>valmax){
				valmax = val;
				hpredict = h;
			}
		}
		return hpredict;
	}
	
	protected double valueOf(X x, Integer y, H h, double[][] w) {
		// compute <w[y], Psi(x,h)>
		return linear.valueOf(w[y], psi(x,h));
	}
	
	/**
	 * Compute the accuracy
	 * @param l list of training samples
	 * @return accuracy
	 */
	protected double accuracy(List<STrainingSample<LatentRepresentation<X, H>, Integer>> l){
		double accuracy = 0;
		int nb = 0;
		for(STrainingSample<LatentRepresentation<X,H>,Integer> ts : l){
			int ypredict = prediction(ts.input);
			if(ts.output == ypredict){	
				nb++;
			}
		}
		accuracy = (double)nb/(double)l.size();
		System.out.println("Accuracy: " + accuracy*100 + " % \t(" + nb + "/" + l.size() +")");
		return accuracy;
	}
	
	public double test(List<STrainingSample<LatentRepresentation<X, H>,Integer>> l) {
		double[] nb = new double[listClass.size()];
		for(STrainingSample<LatentRepresentation<X, H>,Integer> ts : l) {
			nb[ts.output]++;
		}
		System.out.println("Test - class: " + listClass + "\t" + Arrays.toString(nb));
		return accuracy(l);
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
	
	public String toString() {
		return "lssvm_multiclass_fast_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin ;
	}
	
	public void showParameters() {
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train LSSVM - Mosek \tlambda: " + lambda + "\tepochsLatentMax " + epochsLatentMax + "\tepochsLatentMin " + epochsLatentMin); 
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		System.out.println("Multiclass \t dim= " + dim + " \tclasses: " + listClass);
		if(optim == 1) {
			System.out.println("optim " + optim + " \t CCCP - Cutting-Plane 1 Slack");
		}
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	public int getEpochsLatentMax() {
		return epochsLatentMax;
	}
	public void setEpochsLatentMax(int epochsLatentMax) {
		this.epochsLatentMax = epochsLatentMax;
	}
	public int getEpochsLatentMin() {
		return epochsLatentMin;
	}
	public void setEpochsLatentMin(int epochsLatentMin) {
		this.epochsLatentMin = epochsLatentMin;
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
	public void setTradeOff(double tradeoff){
		this.tradeoff = tradeoff;
	}
	public void setLossDict(String lossPath){
		
		try {
			ObjectInputStream is;
			is = new ObjectInputStream(new FileInputStream(lossPath));
			this.positiveLossMap = (HashMap<String, Double> ) is.readObject();// 从流中读取User的数据  
			is.close();}
		catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}  
		
	}
}
